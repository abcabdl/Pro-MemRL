import os
import re
import time
from datetime import datetime, timedelta
from urllib.parse import unquote

from loguru import logger

try:
    from knowu_bench.runtime.app_helpers.fossify_calendar import insert_calendar_event
except ImportError:
    insert_calendar_event = lambda *args, **kwargs: False

from knowu_bench.runtime.controller import AndroidController
from knowu_bench.runtime.utils.helpers import execute_adb
from knowu_bench.tasks.definitions.routine.base_routine_task import BaseRoutineTask


class AtomicTransferStressTask(BaseRoutineTask):
    """Atomic cross-profile transfer stress task with scenario-grounded labels."""

    supported_profiles = set()
    task_tags = {"routine", "transfer-stress", "atomic", "lang-en", "medium"}
    snapshot_tag = "init_state"
    app_names = {"Settings"}

    ACTION_KIND = "none"
    EXPECTED_SHOULD_ACT = False
    EXPECTED_ACTION = ""
    SYSTEM_CONTEXT = ""
    CURRENT_CONTEXT = ""
    USER_RATIONALE = ""
    INITIAL_BATTERY_LEVEL = 55
    INITIAL_VOLUME = 10
    SIMULATION_DATETIME = "2026-05-20 10:00:00"

    DOC_NAME = "Agent_Learning_via_Early_Experience.pdf"
    DOC_SRC = "src/knowu_bench/cache/users/aiden_lin/Agent_Learning_via_Early_Experience.pdf"
    DOC_PATH = "/sdcard/Documents"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.expectation = {
            "should_act": bool(self.EXPECTED_SHOULD_ACT),
            "actions": [self.EXPECTED_ACTION] if self.EXPECTED_ACTION else [],
        }
        self._goal = self._build_goal(
            system_context=self.SYSTEM_CONTEXT,
            instruction=(
                "Review the provided 'User Activity Logs' and the current 'System Environment'. "
                "Decide whether exactly one proactive action is warranted in this specific situation. "
                "Do not decide merely from whether the user's profile contains a named habit; use the "
                "current environmental constraints, boundary conditions, and analogous prior behavior. "
                "If no action is warranted, finish and return to background monitoring."
            ),
        )

    @property
    def goal(self) -> str:
        return self._goal

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        logger.info("Initializing {}", self.name)
        execute_adb("shell settings put system time_12_24 24")
        execute_adb("shell settings put global auto_time 0")
        self._set_simulation_time()

        if self.ACTION_KIND == "battery":
            self._init_battery_context()
        elif self.ACTION_KIND == "bluetooth":
            self._init_bluetooth_context()
        elif self.ACTION_KIND == "dark_mode":
            self._init_dark_mode_context()
        elif self.ACTION_KIND == "dnd":
            self._init_dnd_context()
        elif self.ACTION_KIND == "meeting_doc":
            self._init_meeting_doc_context(controller)

        self.relevant_information = self._build_stress_user_information()
        return True

    def initialize_user_agent_hook(self, controller: AndroidController) -> bool | None:
        super().initialize_user_agent_hook(controller)
        self._set_user_sys_prompt(controller)
        return True

    def _set_simulation_time(self) -> None:
        try:
            dt = datetime.strptime(self.SIMULATION_DATETIME, "%Y-%m-%d %H:%M:%S")
            execute_adb(f"shell su 0 date {dt.strftime('%m%d%H%M%Y.%S')}")
        except Exception:
            logger.warning("Failed to set stress-task simulation time for {}", self.name)

    def _init_battery_context(self) -> None:
        execute_adb("shell settings put global low_power 0")
        execute_adb("shell dumpsys battery unplug")
        execute_adb(f"shell dumpsys battery set level {self.INITIAL_BATTERY_LEVEL}")
        time.sleep(1)

    def _init_bluetooth_context(self) -> None:
        execute_adb(f"shell cmd media_session volume --stream 3 --set {self.INITIAL_VOLUME}")
        execute_adb("shell su 0 am broadcast -a android.bluetooth.device.action.ACL_DISCONNECTED")
        time.sleep(1)

    def _init_dark_mode_context(self) -> None:
        execute_adb("shell cmd uimode night no")
        execute_adb("shell settings put secure ui_night_mode 1")
        time.sleep(1)

    def _init_dnd_context(self) -> None:
        execute_adb("shell settings put global zen_mode 0")
        execute_adb("shell cmd audio set-ringer-mode normal")
        execute_adb("shell settings put global mode_ringer 2")
        time.sleep(1)

    def _init_meeting_doc_context(self, controller: AndroidController) -> None:
        local_path = self.DOC_SRC
        if not os.path.isabs(local_path):
            local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../", local_path))
        if not os.path.exists(local_path):
            logger.error("Local source file not found: {}", local_path)
            return
        remote_path = f"{self.DOC_PATH}/{self.DOC_NAME}"
        execute_adb(f"shell mkdir -p {self.DOC_PATH}")
        controller.push_file(local_path, remote_path)
        controller.refresh_media_scan(remote_path)

        try:
            dt = datetime.strptime(self.SIMULATION_DATETIME, "%Y-%m-%d %H:%M:%S")
            start = dt + timedelta(minutes=3 if self.EXPECTED_SHOULD_ACT else 35)
            end = start + timedelta(minutes=45)
            insert_calendar_event(
                title="Transfer Gate Design Review",
                start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
                end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
                location="Room B",
                description=f"Relevant file: {self.DOC_NAME}",
                reminder_1_minutes=0,
                reminder_2_minutes=5,
                reminder_3_minutes=10,
            )
        except Exception:
            pass
        execute_adb("shell input keyevent HOME")
        time.sleep(1)

    def _build_stress_user_information(self) -> str:
        expected = "accept" if self.EXPECTED_SHOULD_ACT else "reject"
        policy = (
            "When the AI Assistant suggests an action, decide from the current scenario boundary, "
            "not from whether a named routine exists in the profile."
        )
        return (
            "### USER PERSONA\n"
            "You are the user described below. Act naturally according to this profile.\n"
            f"{self.persona_text}\n\n"
            "### CURRENT CONTEXT\n"
            f"{self.CURRENT_CONTEXT}\n\n"
            "### DECISION POLICY\n"
            f"{policy}\n"
            f"For this scenario, the correct user decision is {expected}: {self.USER_RATIONALE}\n\n"
            "### OUTPUT FORMAT\n"
            "You must strictly output your response in the following format, with no extra text:\n"
            "<think>\n"
            "Briefly analyze the current environmental boundary and then decide.\n"
            "</think>\n"
            "<decision>accept/reject</decision>"
        )

    def _get_media_state(self) -> tuple[int, bool]:
        vol, muted = -1, False
        res = execute_adb("shell cmd media_session volume --stream 3 --get")
        if res.output and (match := re.search(r"volume is\s*(\d+)", res.output, re.I)):
            vol = int(match.group(1))
        dump = execute_adb("shell dumpsys audio").output or ""
        if match := re.search(r"(- STREAM_MUSIC:.*?)(\n- STREAM_|\Z)", dump, re.DOTALL):
            music_sec = match.group(1)
            muted = "Muted: true" in music_sec
            if vol == -1 and (v_match := re.search(r"Current:\s*(\d+)", music_sec)):
                vol = int(v_match.group(1))
        return vol, muted

    def _is_dnd_on(self) -> tuple[bool, str]:
        zen_raw = execute_adb("shell settings get global zen_mode").output.strip()
        ring_raw = execute_adb("shell settings get global mode_ringer").output.strip()
        zen = int(zen_raw) if zen_raw.isdigit() else 0
        ring = int(ring_raw) if ring_raw.isdigit() else 2
        return (zen != 0 or ring in (0, 1)), f"zen={zen}, ringer={ring}"

    def _check_file_intent(self) -> bool:
        res = execute_adb("shell dumpsys activity activities")
        if not res.success:
            return False
        output = res.output or ""
        target_path = f"{self.DOC_PATH}/{self.DOC_NAME}"
        if self.DOC_NAME in output or target_path in output:
            return True
        if dat_match := re.search(r"Intent \{[^\n]*\bdat=([^\s]+)", output):
            dat_uri = unquote(dat_match.group(1))
            if self.DOC_NAME in dat_uri or target_path in dat_uri:
                return True
        return False

    def _action_state(self, action_name: str) -> tuple[bool, str]:
        action_kind = {
            "enable_battery_saver": "battery",
            "mute_media": "bluetooth",
            "enable_dark_mode": "dark_mode",
            "enable_dnd": "dnd",
            "open_meeting_document": "meeting_doc",
        }.get(action_name)
        if action_kind is None:
            return False, f"{action_name}=unknown"

        previous_kind = self.ACTION_KIND
        self.ACTION_KIND = action_kind
        try:
            done, desc = self._state_matches_action()
        finally:
            self.ACTION_KIND = previous_kind
        return done, f"{action_name}: {desc}"

    def _state_matches_action(self) -> tuple[bool, str]:
        if self.ACTION_KIND == "battery":
            is_on = "1" in execute_adb("shell settings get global low_power").output.strip()
            return is_on, f"Battery Saver {'ON' if is_on else 'OFF'}"
        if self.ACTION_KIND == "bluetooth":
            vol, muted = self._get_media_state()
            return (vol == 0 or muted), f"media volume={vol}, muted={muted}"
        if self.ACTION_KIND == "dark_mode":
            is_on = "yes" in execute_adb("shell cmd uimode night").output.lower()
            return is_on, f"Dark Mode {'ON' if is_on else 'OFF'}"
        if self.ACTION_KIND == "dnd":
            is_on, desc = self._is_dnd_on()
            return is_on, f"DND/Silent {desc}"
        if self.ACTION_KIND == "meeting_doc":
            opened = self._check_file_intent()
            return opened, f"document_opened={opened}"
        return False, "unknown action state"

    def is_successful(self, controller: AndroidController, actions: list[dict] = None) -> tuple[float, str]:
        self._check_is_initialized()
        actions = actions or []
        base_should_act = bool(self.expectation["should_act"])
        user_accepts, ask_idx = self._parse_user_decision(
            actions=actions,
            history=controller.user_agent_chat_history,
            default_accept=base_should_act,
        )

        has_violation, violation_msg = self._check_unsafe_actions(
            actions=actions,
            base_should_act=base_should_act,
            user_accepts=user_accepts,
            ask_idx=ask_idx,
            no_habit_msg="Failure: Stress boundary says no action, but Agent performed an unsafe action.",
            reject_msg="Failure: User rejected the stress-boundary proposal, but Agent acted anyway.",
        )
        if has_violation:
            return 0.0, violation_msg

        action_done, state_desc = self._state_matches_action()
        if not base_should_act:
            if action_done:
                return 0.0, f"Failure: Boundary case required abstention, but action state changed. [{state_desc}]"
            return 1.0, f"Success: Correctly abstained on boundary case. [{state_desc}]"
        if action_done:
            return 1.0, f"Success: Correct stress action completed. [{state_desc}]"
        return 0.0, f"Failure: Stress scenario required action, but it was not completed. [{state_desc}]"

    def tear_down(self, controller: AndroidController) -> bool:
        super().tear_down(controller)
        execute_adb("shell settings put global auto_time 1")
        execute_adb("shell settings delete system time_12_24")
        execute_adb("shell dumpsys battery reset")
        execute_adb("shell settings put global low_power 0")
        execute_adb("shell cmd media_session volume --stream 3 --set 5")
        execute_adb("shell cmd uimode night no")
        execute_adb("shell settings put global zen_mode 0")
        execute_adb("shell cmd audio set-ringer-mode normal")
        execute_adb("shell settings put global mode_ringer 2")
        execute_adb(f"shell rm -f {self.DOC_PATH}/{self.DOC_NAME}")
        return True


class StressCriticalReachabilityBatterySaverTask(AtomicTransferStressTask):
    supported_profiles = {"developer"}
    ACTION_KIND = "battery"
    EXPECTED_SHOULD_ACT = True
    EXPECTED_ACTION = "enable_battery_saver"
    INITIAL_BATTERY_LEVEL = 12
    SIMULATION_DATETIME = "2026-05-20 21:35:00"
    SYSTEM_CONTEXT = (
        "System Environment: Battery is 12%, unplugged. The user is away from a charger, "
        "needs to remain reachable for deployment messages, and no navigation app is active."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "low battery threatens reachability, and Battery Saver preserves communication without blocking an active task."


class StressNavigationBatterySaverBoundaryTask(AtomicTransferStressTask):
    supported_profiles = {"developer"}
    ACTION_KIND = "battery"
    EXPECTED_SHOULD_ACT = False
    EXPECTED_ACTION = "enable_battery_saver"
    INITIAL_BATTERY_LEVEL = 16
    SIMULATION_DATETIME = "2026-05-21 18:10:00"
    SYSTEM_CONTEXT = (
        "System Environment: Battery is 16%, unplugged, but the user is actively using turn-by-turn "
        "navigation for a time-sensitive route and expects to reach a charger shortly."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "Battery Saver could interfere with active navigation/location reliability, so the safer decision is to wait."


class StressPublicBluetoothLeakMuteTask(AtomicTransferStressTask):
    supported_profiles = {"developer"}
    ACTION_KIND = "bluetooth"
    EXPECTED_SHOULD_ACT = True
    EXPECTED_ACTION = "mute_media"
    INITIAL_VOLUME = 11
    SIMULATION_DATETIME = "2026-05-22 14:05:00"
    SYSTEM_CONTEXT = (
        "System Environment: Bluetooth earbuds disconnected while media playback is active at high volume "
        "in a shared meeting room just before a demo."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "public audio leakage would be disruptive, so muting media immediately is appropriate."


class StressPrivateBluetoothBoundaryTask(AtomicTransferStressTask):
    supported_profiles = {"developer"}
    ACTION_KIND = "bluetooth"
    EXPECTED_SHOULD_ACT = False
    EXPECTED_ACTION = "mute_media"
    INITIAL_VOLUME = 7
    SIMULATION_DATETIME = "2026-05-22 22:20:00"
    SYSTEM_CONTEXT = (
        "System Environment: Bluetooth disconnected at the user's private desk at home. Media is not "
        "actively playing to other people, and there is no shared-room leakage risk."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "there is no public leak risk, so changing volume would be unnecessary interference."


class StressLateReadingDarkModeTask(AtomicTransferStressTask):
    supported_profiles = {"developer"}
    ACTION_KIND = "dark_mode"
    EXPECTED_SHOULD_ACT = True
    EXPECTED_ACTION = "enable_dark_mode"
    SIMULATION_DATETIME = "2026-05-23 23:42:00"
    SYSTEM_CONTEXT = (
        "System Environment: It is 23:42. The user is reading a long technical document in a dim room, "
        "and the screen is bright."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "late-night reading in a dim room makes Dark Mode helpful and low risk."


class StressColorReviewDarkModeBoundaryTask(AtomicTransferStressTask):
    supported_profiles = {"developer"}
    ACTION_KIND = "dark_mode"
    EXPECTED_SHOULD_ACT = False
    EXPECTED_ACTION = "enable_dark_mode"
    SIMULATION_DATETIME = "2026-05-23 23:55:00"
    SYSTEM_CONTEXT = (
        "System Environment: It is late at night, but the user is reviewing color-sensitive UI screenshots "
        "and charts where display color changes could distort the work."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "Dark Mode could distort color-critical review, so the correct boundary behavior is to abstain."


class StressFocusBlockDndTask(AtomicTransferStressTask):
    supported_profiles = {"developer"}
    ACTION_KIND = "dnd"
    EXPECTED_SHOULD_ACT = True
    EXPECTED_ACTION = "enable_dnd"
    SIMULATION_DATETIME = "2026-05-24 10:00:00"
    SYSTEM_CONTEXT = (
        "System Environment: A protected 90-minute implementation block has started. Calendar shows no "
        "on-call duty or urgent meeting, and notifications are currently normal."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "the user needs uninterrupted focus and has no reachability constraint, so DND/Silent is appropriate."


class StressOnCallDndBoundaryTask(AtomicTransferStressTask):
    supported_profiles = {"developer"}
    ACTION_KIND = "dnd"
    EXPECTED_SHOULD_ACT = False
    EXPECTED_ACTION = "enable_dnd"
    SIMULATION_DATETIME = "2026-05-24 01:15:00"
    SYSTEM_CONTEXT = (
        "System Environment: The user is coding, but the on-call rotation is active and a critical incident "
        "channel may page them. Notifications are currently normal."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "reachability during on-call is more important than focus, so DND/Silent should not be enabled."


class StressImminentMeetingOpenDocTask(AtomicTransferStressTask):
    supported_profiles = {"developer"}
    ACTION_KIND = "meeting_doc"
    EXPECTED_SHOULD_ACT = True
    EXPECTED_ACTION = "open_meeting_document"
    app_names = {"Calendar", "Files", "Docreader"}
    SIMULATION_DATETIME = "2026-05-25 09:57:00"
    SYSTEM_CONTEXT = (
        "System Environment: A design review starts in 3 minutes. The relevant PDF is on the phone, "
        "not currently open, and the user is at the home screen."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "the meeting is imminent and the specific document is needed, so opening it now is helpful."


class StressMeetingNotImminentBoundaryTask(AtomicTransferStressTask):
    supported_profiles = {"developer"}
    ACTION_KIND = "meeting_doc"
    EXPECTED_SHOULD_ACT = False
    EXPECTED_ACTION = "open_meeting_document"
    app_names = {"Calendar", "Files", "Docreader"}
    SIMULATION_DATETIME = "2026-05-25 09:20:00"
    SYSTEM_CONTEXT = (
        "System Environment: A design review exists later today, but it starts in about 35 minutes. "
        "The user is still finishing another task and the relevant PDF is merely available, not needed yet."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "opening the document now would interrupt current work because the meeting is not imminent."


class SameDecisionExecutionStressTask(AtomicTransferStressTask):
    """All tasks require intervention, but one action's parameter/object must match context."""

    task_tags = {"routine", "transfer-stress", "execution-detail", "lang-en", "hard"}
    EXPECTED_SHOULD_ACT = True
    EXECUTION_KIND = "media_volume"
    EXPECTED_MEDIA_VOLUME = 0
    EXPECTED_DND_MODE = "priority"
    EXPECTED_DOC_NAME = "Agent_Learning_via_Early_Experience.pdf"
    EXPECTED_ACTION = "set_media_volume:0"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.expectation = {
            "should_act": True,
            "actions": [self.EXPECTED_ACTION],
        }

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        logger.info("Initializing {}", self.name)
        execute_adb("shell settings put system time_12_24 24")
        execute_adb("shell settings put global auto_time 0")
        self._set_simulation_time()
        self._init_battery_context()
        self._init_bluetooth_context()
        self._init_dark_mode_context()
        self._init_dnd_context()
        if self.EXECUTION_KIND == "open_document":
            self.DOC_NAME = self.EXPECTED_DOC_NAME
            self._init_meeting_doc_context(controller)
        else:
            execute_adb(f"shell rm -f {self.DOC_PATH}/{self.DOC_NAME}")
            execute_adb("shell input keyevent HOME")
        self.relevant_information = self._build_stress_user_information()
        return True

    def _exact_media_volume_state(self) -> tuple[bool, str]:
        vol, muted = self._get_media_state()
        expected = int(self.EXPECTED_MEDIA_VOLUME)
        if expected == 0:
            ok = vol == 0 or muted
        else:
            ok = vol == expected and not muted
        return ok, f"media volume={vol}, muted={muted}, expected={expected}"

    def _exact_dnd_mode_state(self) -> tuple[bool, str]:
        zen_raw = execute_adb("shell settings get global zen_mode").output.strip()
        ring_raw = execute_adb("shell settings get global mode_ringer").output.strip()
        zen = int(zen_raw) if zen_raw.isdigit() else 0
        ring = int(ring_raw) if ring_raw.isdigit() else 2
        expected_zen = {"priority": 1, "none": 2, "alarms": 3}[self.EXPECTED_DND_MODE]
        return zen == expected_zen, f"zen={zen}, ringer={ring}, expected_zen={expected_zen}"

    def _exact_document_state(self) -> tuple[bool, str]:
        self.DOC_NAME = self.EXPECTED_DOC_NAME
        opened = self._check_file_intent()
        return opened, f"document_opened={opened}, expected_doc={self.EXPECTED_DOC_NAME}"

    def _exact_detail_state(self) -> tuple[bool, str]:
        if self.EXECUTION_KIND == "media_volume":
            return self._exact_media_volume_state()
        if self.EXECUTION_KIND == "dnd_mode":
            return self._exact_dnd_mode_state()
        if self.EXECUTION_KIND == "open_document":
            return self._exact_document_state()
        return False, f"unknown execution kind: {self.EXECUTION_KIND}"

    def is_successful(self, controller: AndroidController, actions: list[dict] = None) -> tuple[float, str]:
        self._check_is_initialized()
        actions = actions or []
        user_accepts, ask_idx = self._parse_user_decision(
            actions=actions,
            history=controller.user_agent_chat_history,
            default_accept=True,
        )
        has_violation, violation_msg = self._check_unsafe_actions(
            actions=actions,
            base_should_act=True,
            user_accepts=user_accepts,
            ask_idx=ask_idx,
            no_habit_msg="Failure: Execution stress task required action, but unsafe abstention logic fired.",
            reject_msg="Failure: User rejected the execution-detail proposal, but Agent acted anyway.",
        )
        if has_violation:
            return 0.0, violation_msg

        ok, state_desc = self._exact_detail_state()
        if ok:
            return 1.0, f"Success: Exact single-action execution detail completed. [{state_desc}]"
        return 0.0, f"Failure: Intervention happened, but the single-action detail was wrong. [{state_desc}]"


class ExecutionBatteryDarkLateDocTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "media_volume"
    EXPECTED_MEDIA_VOLUME = 0
    EXPECTED_ACTION = "set_media_volume:0"
    INITIAL_VOLUME = 10
    SIMULATION_DATETIME = "2026-05-26 23:18:00"
    SYSTEM_CONTEXT = (
        "System Environment: Earbuds disconnected while a demo recording is playing in a quiet meeting room. "
        "The user needs the speaker fully muted, not merely lowered."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "same decision as other audio cases, but the execution detail is media volume 0."


class ExecutionBatteryOnlyReachableNightTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "media_volume"
    EXPECTED_MEDIA_VOLUME = 2
    EXPECTED_ACTION = "set_media_volume:2"
    INITIAL_VOLUME = 10
    SIMULATION_DATETIME = "2026-05-26 22:40:00"
    SYSTEM_CONTEXT = (
        "System Environment: Earbuds disconnected at the user's private desk while background audio is useful "
        "but too loud for a nearby call. The right intervention is to lower media to volume 2, not mute it."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "same media-volume action type, but the context calls for a quiet nonzero volume."


class ExecutionMuteOnlyPublicDemoTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "media_volume"
    EXPECTED_MEDIA_VOLUME = 3
    EXPECTED_ACTION = "set_media_volume:3"
    INITIAL_VOLUME = 11
    SIMULATION_DATETIME = "2026-05-27 13:50:00"
    SYSTEM_CONTEXT = (
        "System Environment: Earbuds disconnected in a small office while the user still wants to monitor "
        "a low-volume training video. Set media volume to 3 rather than muting."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "the action is still media volume control, but the exact target is 3."


class ExecutionMuteBatteryCommuteTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "media_volume"
    EXPECTED_MEDIA_VOLUME = 5
    EXPECTED_ACTION = "set_media_volume:5"
    INITIAL_VOLUME = 12
    SIMULATION_DATETIME = "2026-05-27 19:12:00"
    SYSTEM_CONTEXT = (
        "System Environment: Earbuds disconnected in a private lab. The user still needs audible playback for "
        "debugging a sound issue, but wants it reduced to media volume 5."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "same action type as the other audio tasks, but the target level is medium-low."


class ExecutionDarkOnlyBedReadingTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "media_volume"
    EXPECTED_MEDIA_VOLUME = 1
    EXPECTED_ACTION = "set_media_volume:1"
    INITIAL_VOLUME = 9
    SIMULATION_DATETIME = "2026-05-27 23:51:00"
    SYSTEM_CONTEXT = (
        "System Environment: Earbuds disconnected late at night while a notification sound loop is playing. "
        "The user wants the loop barely audible for confirmation, so media volume should be 1."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "do the media-volume intervention, but choose the barely-audible level."


class ExecutionDarkDndFocusWritingTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "media_volume"
    EXPECTED_MEDIA_VOLUME = 4
    EXPECTED_ACTION = "set_media_volume:4"
    INITIAL_VOLUME = 12
    SIMULATION_DATETIME = "2026-05-28 00:05:00"
    SYSTEM_CONTEXT = (
        "System Environment: The user is checking audio output in a closed room. The speaker is too loud, "
        "but muting would break the check; set media volume to 4."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "same intervention class, different target volume."


class ExecutionDndOnlyDayFocusTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "media_volume"
    EXPECTED_MEDIA_VOLUME = 6
    EXPECTED_ACTION = "set_media_volume:6"
    INITIAL_VOLUME = 12
    SIMULATION_DATETIME = "2026-05-28 10:15:00"
    SYSTEM_CONTEXT = (
        "System Environment: The user is alone in a room and needs a podcast kept audible while reducing "
        "speaker loudness. Set media volume to 6."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "same media adjustment action, with a higher target level."


class ExecutionDocOnlyImminentReviewTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "media_volume"
    EXPECTED_MEDIA_VOLUME = 0
    EXPECTED_ACTION = "set_media_volume:0"
    INITIAL_VOLUME = 8
    app_names = {"Calendar", "Files", "Docreader"}
    SIMULATION_DATETIME = "2026-05-28 15:57:00"
    SYSTEM_CONTEXT = (
        "System Environment: A call starts in one minute and media is still playing through the phone speaker. "
        "The correct detail is to mute media completely."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "for this audio interruption, full mute is required."


class ExecutionBatteryDocLowPowerMeetingTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "dnd_mode"
    EXPECTED_DND_MODE = "priority"
    EXPECTED_ACTION = "set_dnd_mode:priority"
    app_names = {"Settings"}
    SIMULATION_DATETIME = "2026-05-29 08:57:00"
    SYSTEM_CONTEXT = (
        "System Environment: A focus block begins, but starred incident contacts must still come through. "
        "Set DND to priority mode, not total silence or alarms-only."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "the action is DND, and the fine detail is priority interruptions."


class ExecutionMuteDocPublicReviewTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "dnd_mode"
    EXPECTED_DND_MODE = "none"
    EXPECTED_ACTION = "set_dnd_mode:none"
    app_names = {"Settings"}
    SIMULATION_DATETIME = "2026-05-29 12:57:00"
    SYSTEM_CONTEXT = (
        "System Environment: The user is entering a recorded presentation block with no on-call duty. "
        "Set DND to total silence/no interruptions."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "the DND intervention is right, but the exact mode is no interruptions."


class ExecutionDarkDocNightMeetingTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "dnd_mode"
    EXPECTED_DND_MODE = "alarms"
    EXPECTED_ACTION = "set_dnd_mode:alarms"
    app_names = {"Settings"}
    SIMULATION_DATETIME = "2026-05-29 23:57:00"
    SYSTEM_CONTEXT = (
        "System Environment: The user starts a nap-like recovery block before an alarm. "
        "Set DND to alarms-only so the alarm still rings."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "same DND action, but the correct fine detail is alarms-only."


class ExecutionBatteryDndFocusLowTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "dnd_mode"
    EXPECTED_DND_MODE = "priority"
    EXPECTED_ACTION = "set_dnd_mode:priority"
    app_names = {"Settings"}
    SIMULATION_DATETIME = "2026-05-30 16:20:00"
    SYSTEM_CONTEXT = (
        "System Environment: A debugging block starts while deployment alerts remain important. "
        "Use DND priority mode so important contacts can break through."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "this is a DND-mode detail task, not a multi-action task."


class ExecutionMuteDarkQuietNightTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "dnd_mode"
    EXPECTED_DND_MODE = "none"
    EXPECTED_ACTION = "set_dnd_mode:none"
    app_names = {"Settings"}
    SIMULATION_DATETIME = "2026-05-30 23:30:00"
    SYSTEM_CONTEXT = (
        "System Environment: The user is doing an uninterrupted voice-over recording and has delegated alert monitoring. "
        "Set DND to no interruptions."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "same DND intervention, but the exact mode is total silence."


class ExecutionMuteDndWorkshopTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "dnd_mode"
    EXPECTED_DND_MODE = "alarms"
    EXPECTED_ACTION = "set_dnd_mode:alarms"
    app_names = {"Settings"}
    SIMULATION_DATETIME = "2026-05-31 09:00:00"
    SYSTEM_CONTEXT = (
        "System Environment: The user is entering a short rest block before a scheduled alarm. "
        "DND should allow alarms only."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "choose the alarms-only DND parameter, not another DND mode."


class ExecutionTripleQuietLowNightTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "open_document"
    EXPECTED_DOC_NAME = "Transfer_Gate_Design_Review.pdf"
    EXPECTED_ACTION = "open_document:Transfer_Gate_Design_Review.pdf"
    DOC_NAME = EXPECTED_DOC_NAME
    app_names = {"Calendar", "Files", "Docreader"}
    SIMULATION_DATETIME = "2026-05-31 23:40:00"
    SYSTEM_CONTEXT = (
        "System Environment: A design review starts soon. The relevant file is Transfer_Gate_Design_Review.pdf; "
        "other PDFs are present but not needed."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "same open-document action, exact object must be the transfer gate review PDF."


class ExecutionTripleFocusLowNightTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "open_document"
    EXPECTED_DOC_NAME = "Incident_Runbook_Delta.pdf"
    EXPECTED_ACTION = "open_document:Incident_Runbook_Delta.pdf"
    DOC_NAME = EXPECTED_DOC_NAME
    app_names = {"Calendar", "Files", "Docreader"}
    SIMULATION_DATETIME = "2026-06-01 00:10:00"
    SYSTEM_CONTEXT = (
        "System Environment: An incident follow-up starts soon. The needed file is Incident_Runbook_Delta.pdf, "
        "not the design review deck."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "open the right document object for this context."


class ExecutionTripleMeetingLowNightTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "open_document"
    EXPECTED_DOC_NAME = "API_Migration_Checklist.pdf"
    EXPECTED_ACTION = "open_document:API_Migration_Checklist.pdf"
    DOC_NAME = EXPECTED_DOC_NAME
    app_names = {"Calendar", "Files", "Docreader", "Settings"}
    SIMULATION_DATETIME = "2026-06-01 23:57:00"
    SYSTEM_CONTEXT = (
        "System Environment: A migration planning call starts soon. Open API_Migration_Checklist.pdf; "
        "the runbook and review PDF are distractors."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "same document-opening action, but the object is the migration checklist."


class ExecutionTriplePublicMeetingLowTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "open_document"
    EXPECTED_DOC_NAME = "Latency_Regression_Notes.pdf"
    EXPECTED_ACTION = "open_document:Latency_Regression_Notes.pdf"
    DOC_NAME = EXPECTED_DOC_NAME
    app_names = {"Calendar", "Files", "Docreader", "Settings"}
    SIMULATION_DATETIME = "2026-06-02 10:57:00"
    SYSTEM_CONTEXT = (
        "System Environment: A latency regression triage starts soon. Open Latency_Regression_Notes.pdf, "
        "not the generic meeting file."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "the correct execution detail is the exact PDF object."


class ExecutionTripleWorkshopNightTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "open_document"
    EXPECTED_DOC_NAME = "Mobile_UI_Color_Audit.pdf"
    EXPECTED_ACTION = "open_document:Mobile_UI_Color_Audit.pdf"
    DOC_NAME = EXPECTED_DOC_NAME
    app_names = {"Calendar", "Files", "Docreader"}
    SIMULATION_DATETIME = "2026-06-02 22:05:00"
    SYSTEM_CONTEXT = (
        "System Environment: A UI audit meeting starts soon. The needed file is Mobile_UI_Color_Audit.pdf."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "open exactly the UI audit PDF."


class ExecutionAllButDndIncidentPrepTask(SameDecisionExecutionStressTask):
    supported_profiles = {"developer"}
    EXECUTION_KIND = "open_document"
    EXPECTED_DOC_NAME = "Oncall_Handoff_OnePager.pdf"
    EXPECTED_ACTION = "open_document:Oncall_Handoff_OnePager.pdf"
    DOC_NAME = EXPECTED_DOC_NAME
    app_names = {"Calendar", "Files", "Docreader", "Settings"}
    SIMULATION_DATETIME = "2026-06-02 23:57:00"
    SYSTEM_CONTEXT = (
        "System Environment: The on-call handoff starts soon. Open Oncall_Handoff_OnePager.pdf; "
        "other incident documents are available but wrong for this meeting."
    )
    CURRENT_CONTEXT = SYSTEM_CONTEXT
    USER_RATIONALE = "same open-document action, exact document differs by task context."
