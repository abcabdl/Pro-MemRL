'''
This file provide three distinct components:
1. Environment: The platform which we are interacting with. (PC and Android)
2. Trigger: How we will actually execute the command.
3. Agent: Get the observation from the environment and generate the action.
'''
import io
import os
import json
import time
import base64
import asyncio
import logging
import traceback
import threading
import subprocess
from typing import Iterable, Literal, Optional, Dict, List
from urllib.parse import quote_plus
from types import SimpleNamespace

import colorlog
from PIL import Image
from codelinker import CodeLinker, CodeLinkerConfig, EventProcessor, EventSink
from codelinker.models import SEvent, ChannelTag


from channels import sc
from agentmodule import ActionListener, Executor, ActivityWatchClient
from prompt import SYSTEM_PROMPT
from constant import AgentResponse
from constant import MAX_TRANSFER_SIZE, TIMEOUT, BUFFER_SIZE
from eadp import (
    DecisionContext,
    DualState,
    DynamicCommitmentConfig,
    DynamicCommitmentMapper,
    EventRecord,
    InternalGenerationSignal,
    LearnableEstimatorConfig,
    LearnableSigmoidEstimator,
    RUNTIME_FEEDBACK,
    SignalEstimationLayer,
    SignalEstimationLayerConfig,
    resolve_operation,
)

from register import ToolRegister
from memrl import ProactiveMemRLRuntime, fuse_decision
from personalization import (
    PersonaAwareSimulator,
    PersonaRegistry,
    PersonalizedPlanner,
    UserModel,
    default_state_path,
    load_runtime_state,
    save_runtime_state,
)
toolreg = ToolRegister()
img_base64 = None

# Set the logger format.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = colorlog.ColoredFormatter(
    fmt='%(log_color)s%(levelname)s - %(name)s - %(message)s',
                            log_colors={
                                'DEBUG':    'white',
                                'INFO':     'green',
                                'WARNING':  'yellow',
                                'ERROR':    'red',
                                'CRITICAL': 'red,bg_white',
                            })
# formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# load information.
default_cfg_file = os.path.join(os.path.dirname(__file__), '..', 'private.toml')
if not os.path.exists(default_cfg_file):
    default_cfg_file = os.path.join(os.path.dirname(__file__), 'private.toml')

CL_CFGFILE = os.getenv(key = 'CODELINKER_CFG',
                    default = default_cfg_file)

codelinker_config = CodeLinkerConfig.from_toml(CL_CFGFILE)
codelinker_config.request.default_completions_model = "activeagent"
codelinker_config.request.use_cache = False
codelinker_config.request.save_completions = True

clinker = CodeLinker(config = codelinker_config)
eventSink = EventSink(sinkChannels=sc,logger=logger)

class BasicComponent(EventProcessor):
    def __init__(self,name:str):
        super().__init__(name = name,
                        sink = eventSink)
        self.listen(sc.setup)(self.setup)
        self.cl = clinker

    def gather(self,
            tags: ChannelTag | Iterable[ChannelTag] | None = None,
            return_dumper:Literal['identity','json'] = 'identity') -> str | Iterable[dict]:
        messages = super().gather(tags = tags,return_dumper = 'identity')
        match return_dumper:
            case 'identity':
                return messages
            case 'json':
                for msg in messages:
                    o = msg['content']
                    if isinstance(o,SEvent):
                        msg['content'] = json.dumps({
                            "Time": o.time,
                            "Source": o.source,
                            "Tags": o.tags,
                            "Event": o.content
                        },ensure_ascii=False)
                return messages
            case __:
                raise ValueError(f"return_dumper should be 'identity' or 'json', but got {return_dumper}")

class AndroidEnv(BasicComponent):
    def __init__(self, *,
                server_host:str = '0.0.0.0',
                server_port:int = 9999,
                name = "AndroidEnv",):
        """
        Args:
            server_host (str, optional): the IP of the socket connection. Defaults to '0.0.0.0'.
            server_port (int, optional): the port of the socket connection. Defaults to 9999.
            name (str, optional): The name of the environment. Defaults to "AndroidEnv".
        """
        super().__init__(name)

        self.client_count: int = 0
        self.server_host : str = server_host
        self.server_port : int = server_port

        complete_tools:List[Dict] = toolreg.get_all_tools_dict()
        self.tools:List[Dict] = [t for t in complete_tools if 'android' in t["name"]]

    async def setup(self):
        """
        For the set up of the android:
        1. Establish a socket connection and wait a client to connect.
        2. Listen on several channels.
        """
        async def run_server():
            async with server:
                await server.serve_forever()

        self.logger.info("Initializing Android Environment...")
        self.add(sc.agent.operations, content = json.dumps(self.tools), silent = True)

        logger.info("Android socket waiting for connection.")
        server = await asyncio.start_server(self.handle_client, self.server_host, self.server_port, limit = MAX_TRANSFER_SIZE)
        addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
        logger.info(f'Serving on {addrs}')

        self.server_task = asyncio.create_task(run_server())

        while self.client_count == 0:
            await asyncio.sleep(0.5)

        logger.info("Android socket connected.")
        logger.info("Env setup done.")

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.client_count += 1
        async def read_data():

            while True:
                try:
                    datalen_bytes = await asyncio.wait_for(
                        reader.read(4),
                        timeout=TIMEOUT)
                except asyncio.TimeoutError:
                    logger.info("Timeout waiting for data length. Closing connection.")
                    continue

                if not datalen_bytes:
                    logger.info("Received empty message. Closing connection.")
                    await asyncio.sleep(1)
                    continue

                datalen:int = int.from_bytes(datalen_bytes, byteorder='big')
                logger.info(f"Received data length: {datalen}")
                msg:bytes = b''

                while datalen > 0:
                    try:
                        data = await asyncio.wait_for(reader.read(min(BUFFER_SIZE, datalen)), timeout=TIMEOUT)
                    except asyncio.TimeoutError:
                        print("Timeout waiting for data chunk. Closing connection.")
                        break

                    if not data:
                        break
                    datalen -= len(data)
                    msg += data

                try:
                    msg_str:str = msg.decode('utf-8')
                except UnicodeDecodeError:
                    logger.error("Failed to decode data chunk.")

                if datalen > 0:
                    logger.error('Data integrity error. Closing connection.')

                try:
                    msg_json:Dict = json.loads(msg_str)
                except json.JSONDecodeError:
                    logger.error("Failed to decode JSON message.")
                    self.logger.error(msg_str)
                    continue

                logger.info(f"Receive Msg Type: {msg_json['type']}")
                logger.info(f'Received msg Down.')

                match msg_json["type"]:
                        case "act_error":
                            logger.error(msg_json["act_error"])
                        case "act_ret":
                            if "screenshot" in msg_json["act_ret"] and len(msg_json["act_ret"]["screenshot"]) > 0:
                                global img_base64
                                img_base64 = msg_json["act_ret"].pop("screenshot")

                                img_data = base64.b64decode(img_base64)

                                with open("screenshot.jpeg", "wb") as f:
                                    f.write(img_data)
                                img = Image.open(io.BytesIO(img_data))
                                img.save("screenshot.jpeg")
                            logger.debug(msg_json["act_ret"])
                        case __:
                            logger.info(msg_json[msg_json["type"]])

                msg_str:str = json.dumps(msg_json)
                self.add(sc.observation, msg_str)

        async def write_data():
            data_event:SEvent = self.get(sc.android.write)
            data_str:str = data_event.content
            send_msg:bytes = data_str.encode(encoding = 'utf-8')
            writer.write(len(send_msg).to_bytes(4, byteorder = 'big'))
            writer.write(send_msg)
            await writer.drain()
            logger.info('<Write complete>')

        try:
            read_task = asyncio.create_task(read_data())
            # listen to the write event.
            self.listen(sc.android.write)(write_data)
            await asyncio.gather(read_task)

        except Exception as e:
            logger.error(f"Error in main_process: {e}")
            logger.error(traceback.format_exc())

        finally:
            read_task.cancel()
            writer.close()
            await writer.wait_closed()
            self.client_count -= 1

class PCEnv(BasicComponent):
    def __init__(self, *,
                aw_client:ActivityWatchClient,
                chrome_apps:List[str],
                interval_seconds:int = 15,
                watched_path:List[str] = [],
                name:str = 'PCEnv',
                ):
        """
        Args:
            aw_client (ActivityWatchClient): The client to let us monitor the PC.
            chrome_apps (List[str]): the chromes that you want to monitor( We can't get rid of this :( )
            interval_seconds (int, optional): The pause time between two interactions. Defaults to 15 [seconds].
            name (str, optional): the name of the environment. Defaults to 'PCEnv'.
        """
        super().__init__(name)
        self.aw_client = aw_client
        self.chrome_apps = chrome_apps
        self.interval_seconds = interval_seconds

        self.action_listener = ActionListener(
            aw_client = aw_client,
            chrome_apps = chrome_apps,
            interval_seconds = interval_seconds,
            watched_path=watched_path)

        self.executor = Executor()

        complete_tools = toolreg.get_all_tools_dict()
        self.tools = [t for t in complete_tools if 'android' not in t["name"]]

    async def setup(self):
        self.logger.info("Initializing PC Environment...")

        def start_local_server():
            try:
                subprocess.run(['python', 'main.py'])
            except:
                subprocess.run(['python3', 'main.py'])

        # We set up the uvicorn in another thread, so we don't have to open to terminal.
        self.thread = threading.Thread(target = start_local_server, daemon=True)
        self.thread.start()
        self.logger.info("Local server established.")

        self.add(sc.agent.operations, content = json.dumps(self.tools), silent = True)
        self.listen(sc.pc.notify)(self.execute)
        self.action_listener.start()
        read_task = asyncio.create_task(self.read_data())
        self.logger.info("PC Environment Initialized. Action Listener running...")

        await asyncio.gather(read_task)

    async def read_data(self):

        await asyncio.sleep(self.interval_seconds)

        while True:
            data:Dict = self.action_listener.send_data()
            async with self.get_tag_lock(sc.activity):
                self.add(sc.observation, content = json.dumps(data,ensure_ascii=False))
            await asyncio.sleep(self.interval_seconds)

    async def execute(self):
        operation:str = self.get(sc.agent.execute).content

        if operation in (None, "", "null", "nop"):
            return

        current_event:str = self.get(sc.observation).content
        proposal:str = self.get(sc.agent.propose).content
        proposal_json:Dict = json.loads(proposal)

        exec_args = {"events": current_event, "func_call": operation}
        self.executor.receive(proposal_json, exec_args)
        self.executor.send()

class DemoAgent(BasicComponent):
    def __init__(self,*,
                env:Literal["PC","Mobile"],
                name:str = "ActiveAgent"):
        """
        Args:
            env (Literal['PC','Mobile']): Whether we are on PC or the Mobile.
            name (str, optional): The name of the agent. Defaults to "ActiveAgent".
        """
        super().__init__(name)
        self.env:str = env
        # Demo-friendly mode: make proactive proposals easier to trigger.
        self.aggressive_demo: bool = os.getenv("ACTIVEAGENT_AGGRESSIVE_DEMO", "false").lower() in ("1", "true", "yes", "on")
        self.mapper = DynamicCommitmentMapper(config=DynamicCommitmentConfig())
        self.signal_layer = SignalEstimationLayer(
            config=SignalEstimationLayerConfig(),
            feedback_memory=self.mapper.feedback_memory,
            flow_estimator=LearnableSigmoidEstimator(LearnableEstimatorConfig(feature_dim=7, seed=123)),
            risk_estimator=LearnableSigmoidEstimator(LearnableEstimatorConfig(feature_dim=8, seed=456)),
        )
        self.personalization_enabled = (os.getenv("ACTIVEAGENT_PERSONALIZATION", "true").lower() in ("1", "true", "yes", "on"))
        self.personalization_state_path = os.getenv("ACTIVEAGENT_PERSONALIZATION_STATE", str(default_state_path()))
        self.default_persona_id = os.getenv("ACTIVEAGENT_DEFAULT_PERSONA", "persona_00")
        self.persona_registry: PersonaRegistry | None = None
        self.personalized_planner: PersonalizedPlanner | None = None
        self.user_model: UserModel | None = None
        self.memrl_enabled = os.getenv("ACTIVEAGENT_MEMRL_ENABLED", "false").lower() in ("1", "true", "yes", "on")
        self.memrl_state_dir = os.getenv(
            "ACTIVEAGENT_MEMRL_STATE_DIR",
            os.path.join(os.path.dirname(__file__), "memrl", "state"),
        )
        self.memrl_bootstrap = os.getenv(
            "ACTIVEAGENT_MEMRL_BOOTSTRAP",
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "dataset",
                "agent_data",
                "memrl_episode_bundle",
                "memrl_episodes.jsonl",
            ),
        )
        self.memrl_runtime: ProactiveMemRLRuntime | None = None
        self._load_personalization()
        self._load_memrl_runtime()
        self._load_gate_checkpoint()

    @property
    def memory(self):
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    @staticmethod
    def _is_nullish(value: Optional[str]) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and value.strip().lower() in ("", "null", "none"):
            return True
        return False

    @staticmethod
    def _clamp_01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _load_gate_checkpoint(self) -> None:
        ckpt_path = os.getenv(
            "ACTIVEAGENT_EADP_CHECKPOINT",
            os.path.join(os.path.dirname(__file__), "eadp", "checkpoints", "hybrid_gate_checkpoint.json"),
        )
        if not ckpt_path or not os.path.exists(ckpt_path):
            self.logger.info("EADP checkpoint not found, using default gate parameters.")
            return
        try:
            payload = json.loads(open(ckpt_path, "r", encoding="utf-8").read())
            mapper_cfg = payload.get("mapper_config", {})
            for key in (
                "r0",
                "alpha_flow",
                "alpha_epistemic",
                "alpha_reject",
                "beta_stuck",
                "beta_risk",
                "epsilon_high_threshold",
                "epsilon_low_threshold",
                "risk_high_threshold",
                "risk_low_threshold",
            ):
                if key in mapper_cfg:
                    setattr(self.mapper.config, key, float(mapper_cfg[key]))

            feedback_cfg = payload.get("feedback_memory", {})
            if "decay_lambda" in feedback_cfg:
                self.mapper.feedback_memory.config.decay_lambda = self._clamp_01(float(feedback_cfg["decay_lambda"]))
            if "horizon" in feedback_cfg:
                self.mapper.feedback_memory.config.horizon = max(1, int(feedback_cfg["horizon"]))

            flow_cfg = payload.get("flow_estimator", {})
            risk_cfg = payload.get("risk_estimator", {})
            flow_weights = flow_cfg.get("weights")
            risk_weights = risk_cfg.get("weights")
            if isinstance(flow_weights, list) and len(flow_weights) == len(self.signal_layer.flow_estimator.weights):
                self.signal_layer.flow_estimator.weights = self.signal_layer.flow_estimator.weights * 0.0 + flow_weights
            if isinstance(risk_weights, list) and len(risk_weights) == len(self.signal_layer.risk_estimator.weights):
                self.signal_layer.risk_estimator.weights = self.signal_layer.risk_estimator.weights * 0.0 + risk_weights
            if "bias" in flow_cfg:
                self.signal_layer.flow_estimator.bias = float(flow_cfg["bias"])
            if "bias" in risk_cfg:
                self.signal_layer.risk_estimator.bias = float(risk_cfg["bias"])

            self.logger.info(f"EADP checkpoint loaded: {ckpt_path}")
        except Exception as exc:
            self.logger.warning(f"Failed to load EADP checkpoint {ckpt_path}: {exc}")

    def _load_personalization(self) -> None:
        if not self.personalization_enabled:
            return
        try:
            self.persona_registry = PersonaRegistry.load()
            self.personalized_planner = PersonalizedPlanner()
            state = load_runtime_state(self.personalization_state_path)
            persona_id = str(state.get("persona_id") or self.default_persona_id)
            if persona_id not in self.persona_registry.personas:
                persona_id = sorted(self.persona_registry.personas.keys())[0]
            persona = self.persona_registry.get_persona(persona_id)
            rubric_by_domain = self.persona_registry.rubrics.get(persona_id, {})
            user_model_state = state.get("user_model_state", {}) if isinstance(state.get("user_model_state"), dict) else {}
            self.user_model = UserModel.from_dict(user_model_state, persona=persona, rubric_by_domain=rubric_by_domain) if user_model_state else UserModel(persona=persona, rubric_by_domain=rubric_by_domain)
        except Exception as exc:
            self.logger.warning(f"Failed to initialize personalization: {exc}")
            self.persona_registry = None
            self.personalized_planner = None
            self.user_model = None

    def _load_memrl_runtime(self) -> None:
        if not self.memrl_enabled:
            return
        try:
            alpha = float(os.getenv("ACTIVEAGENT_MEMRL_ALPHA", "0.12"))
            topk = int(os.getenv("ACTIVEAGENT_MEMRL_TOPK", "8"))
            sim_threshold = float(os.getenv("ACTIVEAGENT_MEMRL_SIM_THRESHOLD", "0.18"))
            self.memrl_runtime = ProactiveMemRLRuntime(alpha=alpha, topk=topk, sim_threshold=sim_threshold)
            snapshot_path = os.path.join(self.memrl_state_dir, "memrl_snapshot.jsonl")
            if os.path.exists(snapshot_path):
                self.memrl_runtime.load(self.memrl_state_dir)
                self.logger.info(f"MemRL snapshot loaded: {snapshot_path}")
            elif self.memrl_bootstrap and os.path.exists(self.memrl_bootstrap):
                self.memrl_runtime.warm_start(self.memrl_bootstrap)
                self.memrl_runtime.save(self.memrl_state_dir)
                self.logger.info(f"MemRL bootstrap loaded: {self.memrl_bootstrap}")
            else:
                self.logger.info("MemRL enabled but no bootstrap or snapshot was found.")
        except Exception as exc:
            self.logger.warning(f"Failed to initialize MemRL runtime: {exc}")
            self.memrl_runtime = None

    @staticmethod
    def _extract_latest_observation(obs: List[Dict]) -> Dict:
        if len(obs) == 0:
            return {}
        try:
            wrapped = json.loads(obs[-1]["content"])
            latest = wrapped.get("Event", {})
            if isinstance(latest, str):
                return json.loads(latest)
            if isinstance(latest, dict):
                return latest
        except Exception:
            return {}
        return {}

    def _observation_to_event_records(self, latest_obs: Dict) -> List[EventRecord]:
        events: List[EventRecord] = []
        base_time = float(latest_obs.get("timestamp", 0.0) or 0.0)

        for idx, item in enumerate(latest_obs.get("apps") or []):
            if not isinstance(item, dict):
                continue
            t = float(item.get("timestamp", base_time + idx))
            data = item.get("data", {}) if isinstance(item.get("data"), dict) else {}
            app = str(data.get("app", "unknown"))
            title = data.get("title", "")
            if isinstance(title, list):
                title = " | ".join([str(x) for x in title[:2] if x])
            events.append(EventRecord(time=t, event=f"app={app}; title={title}"))

        for idx, item in enumerate(latest_obs.get("status") or []):
            if not isinstance(item, dict):
                continue
            t = float(item.get("timestamp", base_time + len(events) + idx))
            status = str(item.get("data", {}).get("status", "unknown"))
            events.append(EventRecord(time=t, event=f"user_status={status}"))

        for idx, item in enumerate(latest_obs.get("hot-keys") or []):
            if not isinstance(item, dict):
                continue
            t = float(item.get("time", base_time + len(events) + idx))
            hotkey = str(item.get("data", {}).get("hot_key", ""))
            if hotkey:
                events.append(EventRecord(time=t, event=f"hotkey={hotkey}"))

        if not events:
            events.append(EventRecord(time=base_time, event="no explicit event details"))
        return events

    def _should_force_proposal(self, latest_obs: Dict) -> bool:
        if not self.aggressive_demo or self.env != "PC":
            return False

        # Avoid noisy proposal while explicitly AFK.
        for item in latest_obs.get("status") or []:
            status = item.get("data", {}).get("status")
            if status == "afk":
                return False

        apps = latest_obs.get("apps") or []
        active_apps = set()
        for app in apps:
            if isinstance(app, dict):
                active_apps.add(app.get("data", {}).get("app"))
        trigger_apps = {"Code.exe", "Code", "msedge.exe", "chrome.exe"}
        return len(active_apps.intersection(trigger_apps)) > 0

    def _extract_signal_payload(self, res: AgentResponse) -> Dict[str, Optional[float]]:
        required = ("f_flow", "d_stuck", "epsilon_agent", "r_risk", "delta_rej", "p_need", "p_accept")
        out: Dict[str, Optional[float]] = {key: None for key in required}
        if res.Signals is None:
            return out
        raw = res.Signals.model_dump() if hasattr(res.Signals, "model_dump") else dict(res.Signals)
        for key in required:
            try:
                value = raw.get(key)
                if value is None:
                    continue
                out[key] = self._clamp_01(float(value))
            except Exception:
                out[key] = None
        return out

    @staticmethod
    def _derive_action_features(pred_task: Optional[str], operation: Optional[str]) -> Dict[str, float]:
        task = "" if pred_task is None else pred_task.lower()
        op = "" if operation is None else operation.lower()
        text = f"{task} {op}".strip()
        high_risk = ("delete", "remove", "overwrite", "pay", "purchase", "transfer", "deploy", "execute", "rename")
        medium_risk = ("install", "config", "change", "modify", "update", "refactor", "migrate")
        if any(k in text for k in high_risk):
            return {"reversible": 0.2, "failure_cost": 0.85, "auth_required": 0.9}
        if any(k in text for k in medium_risk):
            return {"reversible": 0.5, "failure_cost": 0.6, "auth_required": 0.4}
        return {"reversible": 0.85, "failure_cost": 0.2, "auth_required": 0.1}

    def _derive_internal_signal(self, signal_payload: Dict[str, Optional[float]]) -> InternalGenerationSignal:
        p_need = signal_payload.get("p_need")
        p_accept = signal_payload.get("p_accept")
        epsilon = signal_payload.get("epsilon_agent")
        if epsilon is None:
            conf_need = 0.0 if p_need is None else abs(float(p_need) - 0.5) * 2.0
            conf_accept = 0.0 if p_accept is None else abs(float(p_accept) - 0.5) * 2.0
            epsilon = 1.0 - min(1.0, 0.5 * (conf_need + conf_accept))
        epsilon = self._clamp_01(float(epsilon))
        return InternalGenerationSignal(
            generation_entropy=5.0 * epsilon,
            generation_confidence=1.0 - epsilon,
        )

    @staticmethod
    def _detect_domain_from_event_records(event_records: List[EventRecord]) -> str:
        text = " ".join(record.event.lower() for record in event_records)
        coding_tokens = ("code", "vscode", "python", "java", "traceback", "terminal", ".py", ".js", ".java")
        writing_tokens = ("document", "draft", "report", "notes", "markdown", "writing", "essay", "blog")
        coding_hits = sum(1 for token in coding_tokens if token in text)
        writing_hits = sum(1 for token in writing_tokens if token in text)
        if coding_hits > writing_hits and coding_hits > 0:
            return "coding"
        if writing_hits > coding_hits and writing_hits > 0:
            return "writing"
        return "other"

    @staticmethod
    def _event_records_to_observations(event_records: List[EventRecord]) -> List[Dict[str, str | float]]:
        return [{"time": float(item.time), "event": str(item.event)} for item in event_records]

    @staticmethod
    def _to_pipeline_signals(pre_signals: Dict[str, float]) -> Dict[str, float]:
        return {
            "flow": float(pre_signals.get("f_flow", 0.0)),
            "stuck": float(pre_signals.get("d_stuck", 0.0)),
            "need": float(pre_signals.get("p_need", 0.0)),
            "accept": float(pre_signals.get("p_accept", 0.0)),
            "risk": float(pre_signals.get("r_risk", 0.0)),
            "uncertainty": float(pre_signals.get("epsilon_agent", 0.0)),
            "progress": max(0.0, 1.0 - float(pre_signals.get("d_stuck", 0.0))),
            "rejection_memory": float(pre_signals.get("delta_rej", 0.0)),
            "timestamp": float(time.time()),
        }

    @staticmethod
    def _build_candidate_payload(res: AgentResponse) -> Dict[str, Optional[str]]:
        return {
            "purpose": res.Purpose,
            "proactive_task": res.Proactive_Task,
            "response": res.Response,
            "operation": res.Operation,
        }

    def _infer_backbone_level(self, res: AgentResponse) -> int:
        if self._is_nullish(res.Proactive_Task) and self._is_nullish(res.Response):
            return 0
        if self._is_nullish(res.Response):
            return 1
        return 2

    @staticmethod
    def _memory_acceptance_label(simulation_prior: Dict) -> str:
        accept_rate = float(simulation_prior.get("historical_accept_rate", 0.0))
        dismiss_rate = float(simulation_prior.get("historical_dismiss_rate", 0.0))
        annoy_rate = float(simulation_prior.get("historical_annoy_rate", 0.0))
        if accept_rate >= max(dismiss_rate, annoy_rate) and accept_rate >= 0.5:
            return "accept"
        if annoy_rate >= max(accept_rate, dismiss_rate) and annoy_rate >= 0.2:
            return "annoyed"
        if dismiss_rate >= 0.25:
            return "dismiss"
        return "ignore"

    def _merge_simulation_result(self, base_simulation: Dict | None, simulation_prior: Dict) -> Dict:
        payload = dict(base_simulation or {})
        payload.setdefault("acceptance", self._memory_acceptance_label(simulation_prior))
        payload.setdefault("acceptance_confidence", float(simulation_prior.get("historical_accept_rate", 0.0)))
        payload["historical_accept_rate"] = float(simulation_prior.get("historical_accept_rate", 0.0))
        payload["historical_dismiss_rate"] = float(simulation_prior.get("historical_dismiss_rate", 0.0))
        payload["historical_annoy_rate"] = float(simulation_prior.get("historical_annoy_rate", 0.0))
        payload["historical_reject_risk"] = float(simulation_prior.get("historical_reject_risk", 0.0))
        payload["memory_consistency_score"] = (
            float(simulation_prior.get("historical_accept_rate", 0.0))
            - float(simulation_prior.get("historical_dismiss_rate", 0.0))
            - float(simulation_prior.get("historical_annoy_rate", 0.0))
        )
        payload["memory_simulation_prior"] = {
            "historical_accept_rate": payload["historical_accept_rate"],
            "historical_dismiss_rate": payload["historical_dismiss_rate"],
            "historical_annoy_rate": payload["historical_annoy_rate"],
            "support_cases": simulation_prior.get("support_cases", []),
            "risk_cases": simulation_prior.get("risk_cases", []),
        }
        return payload

    @staticmethod
    def _compute_signal_score(pre_signals: Dict[str, float]) -> float:
        return (
            float(pre_signals.get("p_need", 0.0))
            + 0.6 * float(pre_signals.get("p_accept", 0.0))
            - 0.5 * float(pre_signals.get("f_flow", 0.0))
            - 0.7 * float(pre_signals.get("r_risk", 0.0))
            - 0.4 * float(pre_signals.get("delta_rej", 0.0))
        )

    def _save_user_model_state(self, *, pending_intervention: Dict | None = None) -> None:
        payload = {
            "default_user_id": "local_default",
        }
        if self.user_model is not None:
            payload["persona_id"] = self.user_model.persona.persona_id
            payload["user_model_state"] = self.user_model.to_dict()
        if pending_intervention is not None:
            payload["pending_intervention"] = pending_intervention
        save_runtime_state(payload, self.personalization_state_path)

    def _apply_personalized_decision(self, res: AgentResponse, *, level: int, simulation: Dict, decision: Dict, domain: str, persona_id: str | None) -> AgentResponse:
        payload = res.model_dump()
        payload["Persona_ID"] = persona_id
        payload["Domain"] = domain
        payload["Simulation"] = simulation
        payload["Decision"] = decision
        if int(level) <= 0:
            payload["Proactive_Task"] = None
            payload["Response"] = None
            payload["Operation"] = resolve_operation(0, payload.get("Operation"))
        elif int(level) == 1:
            if self._is_nullish(payload.get("Proactive_Task")):
                payload["Proactive_Task"] = "Clarify whether help is wanted right now"
            if self._is_nullish(payload.get("Response")):
                payload["Response"] = "Would a short suggestion help right now?"
            payload["Operation"] = resolve_operation(1, payload.get("Operation"))
        else:
            if self._is_nullish(payload.get("Proactive_Task")):
                payload["Proactive_Task"] = "Provide one focused next step"
            if self._is_nullish(payload.get("Response")):
                payload["Response"] = "I have one focused next step that may help right now."
            payload["Operation"] = resolve_operation(2, payload.get("Operation"))
        return AgentResponse(**payload)

    def _fallback_proposal(self, latest_obs: Dict, purpose: Optional[str]) -> AgentResponse:
        query_seed = ""
        if not self._is_nullish(purpose):
            query_seed = str(purpose)
        else:
            titles: List[str] = []
            for app in latest_obs.get("apps") or []:
                data = app.get("data", {}) if isinstance(app, dict) else {}
                raw_titles = data.get("title", [])
                if isinstance(raw_titles, list):
                    titles.extend([str(t) for t in raw_titles if t])
                elif isinstance(raw_titles, str) and raw_titles.strip():
                    titles.append(raw_titles)
            query_seed = " ".join(titles[:2]).strip()

        if query_seed == "":
            query_seed = "coding workflow help"

        query = quote_plus(query_seed[:120])
        return AgentResponse(
            Purpose=purpose or "The user is actively working in coding/browsing context.",
            Thoughts="Aggressive demo mode is on; provide a lightweight proactive suggestion.",
            Proactive_Task="Offer quick resources for the current task",
            Response="I can quickly search and gather a few useful resources for your current task. Want me to do it?",
            Operation=f"search&query={query}&search_engine=bing",
            Signals={
                "f_flow": 0.45,
                "d_stuck": 0.55,
                "epsilon_agent": 0.35,
                "r_risk": 0.30,
                "delta_rej": 0.0,
                "p_need": 0.65,
                "p_accept": 0.70,
            },
        )

    def _estimate_pre_gate(self, latest_obs: Dict) -> tuple[Dict[str, float], object]:
        event_window = self._observation_to_event_records(latest_obs)
        runtime_flags = RUNTIME_FEEDBACK.recent_flags()
        manual_suppressed = any(
            isinstance(item, dict) and item.get("data", {}).get("status") == "afk"
            for item in (latest_obs.get("status") or [])
        )

        if manual_suppressed:
            entropy = 3.2
            confidence = 0.35
        elif latest_obs.get("apps"):
            entropy = 2.0
            confidence = 0.65
        else:
            entropy = 2.8
            confidence = 0.50

        estimate = self.signal_layer.estimate(
            event_window=event_window,
            internal_signal=InternalGenerationSignal(
                generation_entropy=entropy,
                generation_confidence=confidence,
            ),
            action_features={"reversible": 0.85, "failure_cost": 0.2, "auth_required": 0.1},
            recent_quick_rejects=runtime_flags,
        )

        pre_signals = {
            "f_flow": self._clamp_01(float(estimate.f_flow)),
            "d_stuck": self._clamp_01(float(estimate.d_stuck)),
            "epsilon_agent": self._clamp_01(float(estimate.epsilon_agent)),
            "r_risk": self._clamp_01(float(estimate.r_risk)),
            "delta_rej": self._clamp_01(float(estimate.delta_rej)),
            "p_need": self._clamp_01(float(estimate.p_need)),
            "p_accept": self._clamp_01(float(estimate.p_accept)),
        }

        state = DualState(
            flow_index=pre_signals["f_flow"],
            stuck_index=pre_signals["d_stuck"],
            epistemic_confidence=1.0 - pre_signals["epsilon_agent"],
            need_probability=pre_signals["p_need"],
        )
        decision = self.mapper.map_state(
            state=state,
            context=DecisionContext(
                p_need=pre_signals["p_need"],
                p_accept=pre_signals["p_accept"],
                r_risk=pre_signals["r_risk"],
                epsilon_agent=pre_signals["epsilon_agent"],
                delta_rej=pre_signals["delta_rej"],
                recent_quick_rejects=runtime_flags,
                user_pref_reject=False,
                manual_suppressed=manual_suppressed,
            ),
        )
        return pre_signals, decision

    @staticmethod
    def _gate_instruction(level: int) -> str:
        if int(level) <= 0:
            return (
                "Pre-gate decision is Level 0. Keep silent: set `Proactive_Task`=`null`, "
                "`Response`=`null`, and `Operation`=`null`."
            )
        if int(level) == 1:
            return (
                "Pre-gate decision is Level 1. Only probe/clarify with a short question. "
                "Do not execute tools directly; keep `Operation`=`null`."
            )
        return (
            "Pre-gate decision is Level 2. Provide one concrete proactive suggestion with a valid "
            "operation if feasible."
        )

    def _apply_level_constraint(
        self,
        res: AgentResponse,
        decision: object,
        signals: Dict[str, float],
    ) -> AgentResponse:
        level = int(getattr(decision, "commitment_level", 0))

        payload = res.model_dump()
        payload["Signals"] = signals
        thoughts = str(payload.get("Thoughts", ""))
        payload["Thoughts"] = (
            f"{thoughts}\nPreGateDecision: level={level}, "
            f"R(t)={float(getattr(decision, 'r_value', 0.0)):.4f}, "
            f"tau={float(getattr(decision, 'tau', 1.0)):.4f}."
        ).strip()

        if level == 0:
            payload["Proactive_Task"] = None
            payload["Response"] = None
            payload["Operation"] = resolve_operation(0, payload.get("Operation"))
        elif level == 1:
            if self._is_nullish(payload.get("Proactive_Task")):
                payload["Proactive_Task"] = "Clarify your current need"
            if self._is_nullish(payload.get("Response")):
                payload["Response"] = "Before I act, would you like a quick clarification question?"
            payload["Operation"] = resolve_operation(1, payload.get("Operation"))
        else:
            if self._is_nullish(payload.get("Proactive_Task")):
                payload["Proactive_Task"] = "Provide a focused suggestion"
            if self._is_nullish(payload.get("Response")):
                payload["Response"] = "I have a concrete suggestion that may help right now."
            payload["Operation"] = resolve_operation(2, payload.get("Operation"))

        return AgentResponse(**payload)

    async def setup(self):
        logger.info("Initializing Agent...")
        self.listen(sc.observation)(self.propose)
        logger.info("Agent setup done.")

    async def propose(self):

        if self.get_tag_lock(sc.agent.propose).locked():
            logger.error("Another agent is proposing.")
            return

        async with self.get_tag_lock(sc.agent.propose):
            async with self.get_tag_lock(sc.activity):

                ops_event:SEvent = self.get(sc.agent.operations)
                ops:str = ops_event.content

                obs:Dict = self.gather([sc.observation],return_dumper='json')

                history = obs

                # TODO: Can we add user feedback for PC?

                instructions = "Now analyze the history events and provide a task if you think the user needs your help using the given format. If the user is in an email application and there are no mails, you could first refresh the mail by swipe down using `swipe` tool."
                if self.aggressive_demo and self.env == "PC":
                    instructions += " Demo mode is enabled: when context is meaningful and user is active, prefer providing a lightweight proactive suggestion instead of returning null."

                latest_obs = self._extract_latest_observation(obs)
                event_window = self._observation_to_event_records(latest_obs)
                observations = self._event_records_to_observations(event_window)
                domain = self._detect_domain_from_event_records(event_window)
                pre_signals, pre_decision = self._estimate_pre_gate(latest_obs)
                pipeline_signals = self._to_pipeline_signals(pre_signals)
                pre_level = int(getattr(pre_decision, "commitment_level", 0))
                generation_prior = self.memrl_runtime.retrieve_for_generation(observations, pipeline_signals) if self.memrl_runtime is not None else {
                    "preferred_level": 0,
                    "positive_patterns": [],
                    "negative_patterns": [],
                    "avoid_patterns": [],
                    "used_memory_ids": [],
                }
                instructions += " " + self._gate_instruction(pre_level)

                user_content:str = json.dumps({
                    "Instructions": instructions,
                    "operations": ops,
                    "PreGate": {
                        "commitment_level": pre_level,
                        "reason": str(getattr(pre_decision, "reason", "")),
                        "signals": pre_signals,
                    },
                    "MemoryGenerationPrior": {
                        "preferred_level": generation_prior.get("preferred_level", 0),
                        "generation_recommendation": generation_prior.get("generation_recommendation", {}),
                        "intervene_memory_value": generation_prior.get("intervene_memory_value", 0.0),
                        "abstain_memory_value": generation_prior.get("abstain_memory_value", 0.0),
                        "positive_patterns": generation_prior.get("positive_patterns", []),
                        "negative_patterns": generation_prior.get("negative_patterns", []),
                        "avoid_patterns": generation_prior.get("avoid_patterns", []),
                    },
                })

                if self.env == "Mobile":
                    history = history[-1:]

                global img_base64

                if self.env == 'Mobile' and img_base64 is not None:
                    img = [{
                        "type": "image_url",
                        "image_url":{
                            "url": f"data:image/jpeg;base64,{img_base64}",
                            "detail": "low"
                        }
                    }]
                    img_base64 = None

                else:
                    img = []

                logger.debug('Start Proposing....')

                res = await self.cl.exec(
                    prompt = user_content,
                    return_type = AgentResponse,
                    messages = self.memory + history,
                    images = img
                )

                if pre_level == 2 and self._is_nullish(res.Operation) and self._should_force_proposal(latest_obs):
                    self.logger.info("Aggressive demo fallback activated.")
                    res = self._fallback_proposal(latest_obs, res.Purpose)

                res = self._apply_level_constraint(res, pre_decision, pre_signals)

                candidate = self._build_candidate_payload(res)
                simulation_prior = self.memrl_runtime.retrieve_for_simulation(observations, candidate, pipeline_signals) if self.memrl_runtime is not None else {
                    "historical_accept_rate": 0.0,
                    "historical_dismiss_rate": 0.0,
                    "historical_annoy_rate": 0.0,
                    "historical_reject_risk": 0.0,
                    "support_cases": [],
                    "risk_cases": [],
                    "used_memory_ids": [],
                }

                base_simulation: Dict | None = None
                plan: Dict = {}
                persona_id = self.user_model.persona.persona_id if self.user_model is not None else None
                if (
                    self.persona_registry is not None
                    and self.personalized_planner is not None
                    and self.user_model is not None
                    and domain in {"coding", "writing"}
                ):
                    simulator = PersonaAwareSimulator(self.persona_registry)
                    simulation = simulator.simulate(
                        observations=observations,
                        signals=pipeline_signals,
                        domain=domain,
                        persona_id=self.user_model.persona.persona_id,
                        candidate=candidate,
                        user_model=self.user_model,
                    )
                    base_simulation = simulation.to_dict()
                    plan = self.personalized_planner.decide(
                        signals=pipeline_signals,
                        simulation_result={
                            "total_score": simulation.aggregated_scores.total_score,
                            "acceptance_confidence": simulation.acceptance_confidence,
                        },
                        user_model=self.user_model,
                        domain=domain,
                        observations=observations,
                        proactive_task=res.Proactive_Task,
                    )

                simulation_payload = self._merge_simulation_result(base_simulation, simulation_prior)
                decision_prior = self.memrl_runtime.retrieve_for_decision(observations, candidate, simulation_payload, pipeline_signals) if self.memrl_runtime is not None else {
                    "intervene_memory_value": 0.0,
                    "abstain_memory_value": 0.0,
                    "memory_level_mode": 0,
                    "historical_reject_risk": simulation_payload.get("historical_reject_risk", 0.0),
                    "used_memory_ids": [],
                }
                fused = fuse_decision(
                    signal_score=self._compute_signal_score(pre_signals),
                    backbone_level=int(plan.get("level", self._infer_backbone_level(res))),
                    generation_prior=generation_prior,
                    simulation_result=simulation_payload,
                    decision_prior=decision_prior,
                )
                decision_payload = dict(plan)
                decision_payload.update(
                    {
                        "should_intervene": bool(fused["should_intervene"]),
                        "commitment_level": int(fused["level"]),
                        "fusion_reason": fused["reason"],
                        "signal_score": fused["signal_score"],
                        "simulation_score": fused["simulation_score"],
                        "intervene_memory_value": fused["intervene_memory_value"],
                        "abstain_memory_value": fused["abstain_memory_value"],
                        "historical_reject_risk": fused["historical_reject_risk"],
                        "memory_generation_prior": {
                            "preferred_level": generation_prior.get("preferred_level", 0),
                            "generation_recommendation": generation_prior.get("generation_recommendation", {}),
                            "intervene_memory_value": generation_prior.get("intervene_memory_value", 0.0),
                            "abstain_memory_value": generation_prior.get("abstain_memory_value", 0.0),
                            "positive_patterns": generation_prior.get("positive_patterns", []),
                            "negative_patterns": generation_prior.get("negative_patterns", []),
                            "avoid_patterns": generation_prior.get("avoid_patterns", []),
                        },
                        "memory_decision_prior": {
                            "intervene_memory_value": decision_prior.get("intervene_memory_value", 0.0),
                            "abstain_memory_value": decision_prior.get("abstain_memory_value", 0.0),
                            "memory_level_mode": decision_prior.get("memory_level_mode", 0),
                        },
                    }
                )
                res = self._apply_personalized_decision(
                    res,
                    level=int(fused["level"]),
                    simulation=simulation_payload,
                    decision=decision_payload,
                    domain=domain,
                    persona_id=persona_id,
                )

                if int(fused["level"]) > 0:
                    used_memory_ids = []
                    for bucket in (
                        generation_prior.get("used_memory_ids", []),
                        simulation_prior.get("used_memory_ids", []),
                        decision_prior.get("used_memory_ids", []),
                    ):
                        for memory_id in bucket:
                            if memory_id not in used_memory_ids:
                                used_memory_ids.append(memory_id)
                    online_episode = {
                        "memory_id": f"online-{int(time.time() * 1000)}",
                        "sample_id": f"online-{int(time.time() * 1000)}",
                        "source": "online_runtime",
                        "domain": domain,
                        "observations": observations,
                        "intent_text": " | ".join(
                            part for part in [
                                candidate.get("purpose"),
                                candidate.get("proactive_task"),
                                candidate.get("response"),
                            ] if part
                        ),
                        "candidate": dict(candidate),
                        "simulation": dict(simulation_payload),
                        "decision": {
                            "should_intervene": bool(fused["should_intervene"]),
                            "commitment_level": int(fused["level"]),
                            "risk": "high" if simulation_payload.get("historical_reject_risk", 0.0) > 0.45 else "low",
                            "reason": fused["reason"],
                        },
                        "labels": {
                            "y_need": None,
                            "y_accept": None,
                            "gold_should": None,
                            "gold_level": None,
                            "q_need": pre_signals.get("p_need"),
                            "q_accept": pre_signals.get("p_accept"),
                        },
                        "reward": 0.0,
                        "q_value": 0.0,
                        "q_visits": 0,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    self._save_user_model_state(
                        pending_intervention={
                            "timestamp": time.time(),
                            "signals": pipeline_signals,
                            "candidate": candidate,
                            "domain": domain,
                            "simulation": simulation_payload,
                            "decision": decision_payload,
                            "persona_id": persona_id,
                            "memrl": {
                                "used_memory_ids": used_memory_ids,
                                "episode": online_episode,
                            },
                        }
                    )
                else:
                    self._save_user_model_state()

                self.logger.info(res)
                self.add(sc.agent.propose, content = res.model_dump_json())

                if not self._is_nullish(res.Operation):
                    self.add(sc.agent.execute, res.Operation)
                else:
                    self.add(sc.agent.execute, "nop")

class Trigger(BasicComponent):
    def __init__(self,*,
                env: Literal["PC","Mobile"],
                name:str = "Trigger",
                ):
        """
        Args:
            env (Literal['PC','Mobile']): Whether we are on PC or the Mobile. we send the proposal to different channels.
            name (str, optional): The name of the agent. Defaults to "Trigger".
        """
        super().__init__(name)
        self.env:str = env

    async def setup(self):
        logger.info("Initializing Trigger...")
        self.listen(sc.agent.execute)(self.execute)
        logger.info("Trigger setup done.")

    async def execute(self):
        def reformat_action(tool_description:Optional[str] = 'nop') -> Dict:
            """
            (Android only) Reformat the description from agent to restriced format.

            Args:
                tool_description (str): a string containing the name of the tool and the arguments joined by separator '&'
                Example input: func_name&param1=value1&param2=value2
            """

            nop_action = {
                "type": "action",
                "action": {
                    "nop": {
                        "screenshot": True
                    }
                }
            }

            if tool_description == 'nop':
                return nop_action

            action_json = None

            func_list = tool_description.split('&')

            func_name = func_list[0]
            func_param = func_list[1:]

            try:
                param_dict = {k:v for k,v in [p.split('=') for p in func_param]}
            except:
                param_dict = {}

            # The fucntion name is changed beacuse of the unique function name in the agent. so we manually change this.
            match func_name:
                case 'android_tap_viewId':
                    action_json = {
                        "type": "action",
                        "action": {
                            "tap": param_dict
                        }
                    }

                case 'android_tap_position':
                    action_json = {
                        "type": "action",
                        "action": {
                            "tap": {
                                "coordinates": param_dict
                            }
                        }
                    }

                case 'android_press_viewId':
                    action_json = {
                        "type": "action",
                        "action": {
                            "press": param_dict
                        }
                    }

                case 'android_press_pos':
                    action_json = {
                        "type": "action",
                        "action": {
                            "press": {
                                "coordinates":{
                                    'x' : param_dict["x"],
                                    'y' : param_dict["y"]
                                },
                                "duration": param_dict["duration"]
                            }
                        }
                    }

                case 'android_input':
                    action_json = {
                        "type": "action",
                        "action": {
                            "input": param_dict
                        }
                    }
                    if action_json["action"]["input"]["viewId"] is None:
                        del action_json["action"]["input"]["viewId"]

                case 'android_swipe':
                    action_json = {
                        "type": "action",
                        "action": {
                            "swipe":{
                                "start_coordinates":{
                                    "x": param_dict["start_x"],
                                    "y": param_dict["start_y"],
                                },
                                "end_coordinates":{
                                    "x": param_dict["end_x"],
                                    "y": param_dict["end_y"],
                                },
                                "duration": param_dict["duration"]
                            }
                        }
                    }

                case 'android_back':
                    action_json = {
                        "type": "action",
                        "action": {
                            "back": {}
                        }
                    }

                case 'android_home':
                    action_json = {
                        "type": "action",
                        "action": {
                            "home": {}
                        }
                    }

                case 'android_get_notification':
                    action_json = {
                        "type": "notifications_get_all","notifications_get_all": {}
                    }

                case 'android_add_notification':
                    action_json = {
                        "type": "notifications_add",
                        "notifications_add": param_dict
                    }

                case 'android_get_calendar':
                    action_json = {
                        "type": "calendar_get",
                        "calendar_get": param_dict
                    }

                case 'android_add_calendar':
                    action_json = {
                        "type": "calendar_add",
                        "calendar_add": param_dict
                    }

                case __:
                    self.logger.warning(f"Invalid action {func_name}")
                    return nop_action

            return action_json

        operation:str = self.get(sc.agent.execute).content

        match self.env:
            case 'Mobile':
                action_json:Dict = reformat_action(operation)
                self.add(sc.android.write, content = json.dumps(action_json))

            case 'PC':
                self.add(sc.pc.notify, content = operation)

            case __:
                raise Exception(f"Invalid Environment parameters. {self.env}")
