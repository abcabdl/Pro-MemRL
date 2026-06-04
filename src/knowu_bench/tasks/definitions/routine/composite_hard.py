from __future__ import annotations

import time
import re

from knowu_bench.runtime.app_helpers.mail import get_sent_email_info
from knowu_bench.runtime.app_helpers.mattermost import (
    DEFAULT_PASSWORD,
    MattermostCLI,
    TEAM_NAME,
    USERS,
    get_latest_user_post_after,
    is_mattermost_healthy,
    start_mattermost_backend,
)
from knowu_bench.runtime.controller import AndroidController
from knowu_bench.runtime.utils.helpers import execute_adb
from knowu_bench.runtime.utils.proxy_config import android_proxy_setting_command
from knowu_bench.runtime.utils.routine_time import format_adb_datetime, resolve_routine_datetime
from knowu_bench.tasks.definitions.routine.base_routine_task import BaseRoutineTask


ALARM_DB_PATH = "/data/user_de/0/com.google.android.deskclock/databases/alarms.db"
CLOCK_PKG = "com.google.android.deskclock"


class DeveloperCompositeRoutineMixin:
    @staticmethod
    def _normalize_channel(name: str) -> str:
        return (name or "").strip().lower().replace(" ", "-")

    @staticmethod
    def _extract_keywords(content: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9']+", (content or "").lower())
        return [token for token in dict.fromkeys(tokens) if len(token) >= 3][:8] or ["clocking", "out"]

    @staticmethod
    def _media_muted() -> bool:
        status = execute_adb("shell cmd media_session volume --stream 3 --get")
        audio = execute_adb("shell dumpsys audio")
        return (
            status.success
            and "volume is 0" in (status.output or "").lower()
        ) or "STREAM_MUSIC" in (audio.output or "") and "Current: 0" in (audio.output or "")

    @staticmethod
    def _dark_mode() -> bool:
        status = execute_adb("shell cmd uimode night")
        return status.success and "yes" in (status.output or "").lower()

    def _wait_for_mattermost_ready(self, timeout_sec: int = 45) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if is_mattermost_healthy():
                return True
            time.sleep(2)
        return False

    def _send_alert(self) -> bool:
        try:
            start_mattermost_backend()
            if not self._wait_for_mattermost_ready():
                return False
            cli = MattermostCLI()
            if not cli.login(USERS["sam"], DEFAULT_PASSWORD):
                return False
            try:
                return cli.send_message(self.team_name, self.channel_name, self.alert_msg)
            finally:
                cli.logout()
        except Exception:
            return False

    def _alert_acknowledged(self) -> tuple[bool, str]:
        msg = get_latest_user_post_after(
            start_timestamp=self.start_timestamp,
            channel_name=self.channel_name,
            exclude_message=self.alert_msg,
        )
        if not msg:
            return False, ""
        lowered = msg.strip().lower()
        return any(keyword in lowered for keyword in self.expected_keywords), lowered

    def _clock_out_sent(self) -> tuple[bool, str]:
        msg = get_latest_user_post_after(
            start_timestamp=self.start_timestamp,
            channel_name=self.channel_name,
        )
        if not msg:
            return False, ""
        lowered = msg.strip().lower()
        matched = self.target_content.lower() in lowered or any(
            keyword in lowered for keyword in self.expected_keywords
        )
        return matched, lowered


class LowBatteryMeetingPrepTask(BaseRoutineTask):
    """Composite routine: low battery mitigation plus imminent meeting preparation."""

    task_tags = {"routine", "composite", "system-settings", "calendar", "file_management", "lang-en", "hard"}
    snapshot_tag = "init_state"
    app_names = {"Settings", "Calendar", "Files", "Docreader"}
    supported_profiles = {"user"}

    DOC_NAME = "Agent_Learning_via_Early_Experience.pdf"
    DOC_SOURCE = "src/knowu_bench/cache/users/aiden_lin/Agent_Learning_via_Early_Experience.pdf"
    DOC_PATH = "/sdcard/Documents"
    BATTERY_LEVEL = 13

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.low_battery_habit = self._get_habit("low_battery_saver")
        self.meeting_habit = self._get_habit("pre_meeting_prep")
        self.expectation = {
            "should_act": bool(self.low_battery_habit and self.meeting_habit),
            "actions": ["enable_battery_saver", "open_meeting_file"],
        }
        trigger = self.meeting_habit.get("trigger", {}) if self.meeting_habit else {}
        action = self.meeting_habit.get("action", {}) if self.meeting_habit else {}
        self.doc_name = action.get("file_name", self.DOC_NAME)
        self.doc_source = action.get("file_source", self.DOC_SOURCE)
        self.meeting_title = trigger.get("meeting_title", "Product Review")
        self.simulation_dt = resolve_routine_datetime(
            trigger,
            default_time="09:53:00",
            task_name=self.name,
        )
        self._goal = self._build_goal(
            system_context=(
                f"System Status: Battery Level {self.BATTERY_LEVEL}% and unplugged.\n"
                f"Calendar: '{self.meeting_title}' starts in about 2 minutes.\n"
                "The phone is on the home screen and the user is walking into the meeting room."
            )
        )

    @property
    def goal(self) -> str:
        return self._goal

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        execute_adb("shell settings put global auto_time 0")
        execute_adb("shell settings put system time_12_24 24")
        execute_adb(f"shell su 0 date {format_adb_datetime(self.simulation_dt)}")
        execute_adb("shell settings put global low_power 0")
        execute_adb("shell dumpsys battery unplug")
        execute_adb(f"shell dumpsys battery set level {self.BATTERY_LEVEL}")

        local_path = self.doc_source
        if not local_path.startswith("/") and ":" not in local_path[:3]:
            from pathlib import Path

            local_path = str((Path(__file__).resolve().parents[5] / local_path).resolve())
        remote_path = f"{self.DOC_PATH}/{self.doc_name}"
        execute_adb(f"shell mkdir -p {self.DOC_PATH}")
        controller.push_file(local_path, remote_path)
        controller.refresh_media_scan(remote_path)

        routine_hint = (
            "You HAVE both the low-battery saver routine and the pre-meeting preparation routine."
            if self.expectation["should_act"]
            else "You do NOT have enough established routine evidence for this combined intervention."
        )
        self.relevant_information = self._build_relevant_information(
            current_context=(
                f"It is {self.simulation_dt.strftime('%H:%M')}.\n"
                f"Battery is {self.BATTERY_LEVEL}% and unplugged.\n"
                f"Upcoming meeting: '{self.meeting_title}' starts very soon."
            ),
            routine_status=routine_hint,
            task_specific_detail=(
                f"If accepted, enable Battery Saver and open '{self.doc_name}'. "
                "Doing only one of the two is incomplete."
            ),
        )
        return True

    def initialize_user_agent_hook(self, controller: AndroidController) -> bool | None:
        super().initialize_user_agent_hook(controller)
        self._set_user_sys_prompt(controller)
        return True

    def _document_is_open(self) -> bool:
        status = execute_adb("shell dumpsys activity activities")
        return status.success and self.doc_name in (status.output or "")

    def is_successful(self, controller: AndroidController, actions: list[dict] | None = None) -> tuple[float, str]:
        self._check_is_initialized()
        actions = actions or []
        base_should_act = self.expectation["should_act"]
        user_accepts, ask_idx = self._parse_user_decision(
            actions=actions,
            history=controller.user_agent_chat_history,
            default_accept=base_should_act,
        )
        should_execute = user_accepts if ask_idx != -1 else base_should_act

        has_violation, violation_msg = self._check_unsafe_actions(
            actions=actions,
            base_should_act=base_should_act,
            user_accepts=user_accepts,
            ask_idx=ask_idx,
            no_habit_msg="Failure: Composite action executed without both established routines.",
            reject_msg="Failure: Composite action executed after user rejection.",
        )
        if has_violation:
            return 0.0, violation_msg

        low_power = execute_adb("shell settings get global low_power").output.strip() == "1"
        doc_open = self._document_is_open()

        if not should_execute:
            if low_power or doc_open:
                return 0.0, f"Failure: Agent disturbed user despite no confirmed combined routine. low_power={low_power}, doc_open={doc_open}"
            return 1.0, "Success: Agent correctly abstained from the combined intervention."

        if not low_power and not doc_open:
            return 0.0, "Failure: Agent missed both required actions."
        if not low_power:
            return 0.0, "Failure: Meeting file was handled but Battery Saver was not enabled."
        if not doc_open:
            return 0.0, "Failure: Battery Saver was enabled but the meeting file was not opened."
        return 1.0, "Success: Battery Saver enabled and meeting document opened."

    def tear_down(self, controller: AndroidController) -> bool:
        super().tear_down(controller)
        execute_adb("shell settings put global auto_time 1")
        execute_adb("shell settings delete system time_12_24")
        execute_adb("shell dumpsys battery reset")
        execute_adb("shell settings put global low_power 0")
        execute_adb(f"shell rm -f {self.DOC_PATH}/{self.doc_name}")
        return True


class FridayReportAndWeekendAlarmTask(BaseRoutineTask):
    """Composite routine: send Friday report and disable the next morning weekend alarm."""

    task_tags = {"routine", "composite", "mail", "clock", "lang-en", "hard"}
    snapshot_tag = "init_state"
    app_names = {"Mail", "Clock"}
    supported_profiles = {"user"}

    FILE_NAME = "Weekly_Report.pdf"
    REMOTE_FILE_PATH = f"/sdcard/Documents/{FILE_NAME}"
    MAIL_PACKAGE = "com.gmailclone"
    ALARM_TIME = (7, 30)

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.report_habit = self._get_habit("weekly_report")
        self.weekend_habit = self._get_habit("weekend_sleeper")
        self.expectation = {
            "should_act": bool(self.report_habit and self.weekend_habit),
            "actions": ["send_weekly_report", "disable_weekend_alarm"],
        }
        trigger = self.report_habit.get("trigger", {}) if self.report_habit else {}
        self.recipient = (
            self.report_habit.get("action", {}).get("recipient", "dean@pku.edu.cn")
            if self.report_habit
            else "dean@pku.edu.cn"
        )
        self.simulation_dt = resolve_routine_datetime(
            trigger or {"day_of_week": "Friday", "time_range": ["16:55", "17:05"]},
            default_time="16:59:00",
            task_name=self.name,
        )
        self.bedtime_dt = self.simulation_dt.replace(hour=23, minute=0, second=0)
        self._goal = self._build_goal(
            system_context=(
                f"It is Friday {self.simulation_dt.strftime('%H:%M')}.\n"
                "Weekly_Report.pdf is ready in Documents.\n"
                "There is an enabled everyday 07:30 alarm that would ring tomorrow morning, which is Saturday."
            )
        )

    @property
    def goal(self) -> str:
        return self._goal

    def _run_alarm_sql(self, sql: str) -> str:
        res = execute_adb(f'shell "sqlite3 {ALARM_DB_PATH} \\"{sql}\\""', root_required=True)
        return res.output.strip() if res.success else ""

    def _inject_alarm(self) -> None:
        hour, minute = self.ALARM_TIME
        self._run_alarm_sql(f"DELETE FROM alarm_templates WHERE hour={hour} AND minutes={minute};")
        self._run_alarm_sql(
            "INSERT INTO alarm_templates (hour, minutes, enabled, daysofweek, vibrate, label, ringtone, delete_after_use) "
            f"VALUES ({hour}, {minute}, 1, 127, 1, 'Work', '', 0);"
        )

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        execute_adb("shell settings put global auto_time 0")
        execute_adb("shell settings put system time_12_24 24")
        execute_adb(f"shell su 0 date {format_adb_datetime(self.simulation_dt)}")

        execute_adb("shell mkdir -p /sdcard/Documents")
        execute_adb(f"shell touch {self.REMOTE_FILE_PATH}")
        execute_adb(f"shell chmod 666 {self.REMOTE_FILE_PATH}")
        execute_adb(
            f"shell am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file://{self.REMOTE_FILE_PATH}"
        )

        self._inject_alarm()
        execute_adb(f"shell am force-stop {self.MAIL_PACKAGE}")
        execute_adb(f"shell am force-stop {CLOCK_PKG}")

        routine_hint = (
            "You HAVE both the weekly-report routine and the weekend-sleeper routine."
            if self.expectation["should_act"]
            else "You do NOT have enough established routine evidence for doing both actions."
        )
        self.relevant_information = self._build_relevant_information(
            current_context=(
                f"It is Friday {self.simulation_dt.strftime('%H:%M')}.\n"
                "The weekly report file is ready.\n"
                "Tomorrow is Saturday, and the 07:30 everyday alarm is currently enabled."
            ),
            routine_status=routine_hint,
            task_specific_detail=(
                "If accepted, send the weekly report email and disable the 07:30 weekend alarm. "
                "Both actions are required."
            ),
        )
        return True

    def initialize_user_agent_hook(self, controller: AndroidController) -> bool | None:
        super().initialize_user_agent_hook(controller)
        self._set_user_sys_prompt(controller)
        return True

    def _alarm_enabled(self) -> bool:
        hour, minute = self.ALARM_TIME
        return self._run_alarm_sql(f"SELECT enabled FROM alarm_templates WHERE hour={hour} AND minutes={minute};") == "1"

    def _report_sent(self) -> bool:
        try:
            info = get_sent_email_info()
        except Exception:
            info = None
        if not info:
            return False
        recipient_ok = self.recipient.lower() in str(info.get("to", "")).lower()
        attachment_ok = any(self.FILE_NAME in att.get("name", "") for att in info.get("attachments", []))
        return recipient_ok and attachment_ok

    def is_successful(self, controller: AndroidController, actions: list[dict] | None = None) -> tuple[float, str]:
        self._check_is_initialized()
        actions = actions or []
        base_should_act = self.expectation["should_act"]
        user_accepts, ask_idx = self._parse_user_decision(
            actions=actions,
            history=controller.user_agent_chat_history,
            default_accept=base_should_act,
        )
        should_execute = user_accepts if ask_idx != -1 else base_should_act

        has_violation, violation_msg = self._check_unsafe_actions(
            actions=actions,
            base_should_act=base_should_act,
            user_accepts=user_accepts,
            ask_idx=ask_idx,
            no_habit_msg="Failure: Agent performed a combined report/alarm action without both routines.",
            reject_msg="Failure: Agent performed a combined report/alarm action after user rejection.",
        )
        if has_violation:
            return 0.0, violation_msg

        report_sent = self._report_sent()
        alarm_enabled = self._alarm_enabled()
        alarm_disabled = not alarm_enabled

        if not should_execute:
            if report_sent or alarm_disabled:
                return 0.0, f"Failure: Agent disturbed user. report_sent={report_sent}, alarm_disabled={alarm_disabled}"
            return 1.0, "Success: Agent correctly abstained from the combined routine."

        if not report_sent and not alarm_disabled:
            return 0.0, "Failure: Agent missed both required actions."
        if not report_sent:
            return 0.0, "Failure: Alarm was disabled but weekly report was not sent correctly."
        if not alarm_disabled:
            return 0.0, "Failure: Weekly report was sent but the weekend alarm remains enabled."
        return 1.0, "Success: Weekly report sent and weekend alarm disabled."

    def tear_down(self, controller: AndroidController) -> bool:
        super().tear_down(controller)
        execute_adb("shell settings put global auto_time 1")
        execute_adb("shell settings delete system time_12_24")
        execute_adb(f"shell rm -f {self.REMOTE_FILE_PATH}")
        hour, minute = self.ALARM_TIME
        self._run_alarm_sql(f"DELETE FROM alarm_templates WHERE hour={hour} AND minutes={minute};")
        return True


class QuietHoursBluetoothBatteryTask(DeveloperCompositeRoutineMixin, BaseRoutineTask):
    """Composite routine: prevent audio leakage and save power during a late quiet-hours session."""

    task_tags = {"routine", "composite", "system-settings", "audio", "battery", "lang-en", "hard"}
    snapshot_tag = "init_state"
    app_names = {"Settings"}
    supported_profiles = {"developer"}

    BATTERY_LEVEL = 16

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.bluetooth_habit = self._get_habit("bluetooth_cleanup")
        self.battery_habit = self._get_habit("low_battery_saver")
        self.eye_habit = self._get_habit("night_eye_care")
        self.expectation = {
            "should_act": bool(self.bluetooth_habit and self.battery_habit and self.eye_habit),
            "actions": ["mute_media", "enable_battery_saver", "enable_dark_mode"],
        }
        trigger = self.eye_habit.get("trigger", {}) if self.eye_habit else {}
        self.simulation_dt = resolve_routine_datetime(
            trigger,
            default_time="23:20:00",
            task_name=self.name,
        )
        self._goal = self._build_goal(
            system_context=(
                f"It is {self.simulation_dt.strftime('%H:%M')}.\n"
                f"System Status: Battery Level {self.BATTERY_LEVEL}% and unplugged.\n"
                "Bluetooth headphones just disconnected while media playback is active.\n"
                "The user is in a quiet shared workspace."
            )
        )

    @property
    def goal(self) -> str:
        return self._goal

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        execute_adb("shell settings put global auto_time 0")
        execute_adb("shell settings put system time_12_24 24")
        execute_adb(f"shell su 0 date {format_adb_datetime(self.simulation_dt)}")
        execute_adb("shell settings put global low_power 0")
        execute_adb("shell cmd uimode night no")
        execute_adb("shell dumpsys battery unplug")
        execute_adb(f"shell dumpsys battery set level {self.BATTERY_LEVEL}")
        execute_adb("shell cmd audio set-stream-volume 3 7")
        for _ in range(7):
            execute_adb("shell input keyevent 24")

        routine_hint = (
            "You HAVE bluetooth cleanup, low-battery saver, and late-night eye-care routines."
            if self.expectation["should_act"]
            else "You do NOT have enough established routine evidence for this triple intervention."
        )
        self.relevant_information = self._build_relevant_information(
            current_context=(
                f"It is {self.simulation_dt.strftime('%H:%M')} in a quiet shared workspace.\n"
                f"Battery is {self.BATTERY_LEVEL}% and unplugged.\n"
                "Bluetooth headphones disconnected during active media playback."
            ),
            routine_status=routine_hint,
            task_specific_detail=(
                "If accepted, mute media, enable Battery Saver, and enable Dark Mode. "
                "All three actions are required."
            ),
        )
        return True

    def initialize_user_agent_hook(self, controller: AndroidController) -> bool | None:
        super().initialize_user_agent_hook(controller)
        self._set_user_sys_prompt(controller)
        return True

    def is_successful(self, controller: AndroidController, actions: list[dict] | None = None) -> tuple[float, str]:
        self._check_is_initialized()
        actions = actions or []
        base_should_act = self.expectation["should_act"]
        user_accepts, ask_idx = self._parse_user_decision(
            actions=actions,
            history=controller.user_agent_chat_history,
            default_accept=base_should_act,
        )
        should_execute = user_accepts if ask_idx != -1 else base_should_act

        has_violation, violation_msg = self._check_unsafe_actions(
            actions=actions,
            base_should_act=base_should_act,
            user_accepts=user_accepts,
            ask_idx=ask_idx,
            no_habit_msg="Failure: Agent performed quiet-hours system changes without all required routines.",
            reject_msg="Failure: Agent performed quiet-hours system changes after user rejection.",
        )
        if has_violation:
            return 0.0, violation_msg

        muted = self._media_muted()
        low_power = execute_adb("shell settings get global low_power").output.strip() == "1"
        dark_mode = self._dark_mode()

        if not should_execute:
            if muted or low_power or dark_mode:
                return 0.0, f"Failure: Agent disturbed user. muted={muted}, low_power={low_power}, dark_mode={dark_mode}"
            return 1.0, "Success: Agent correctly abstained from the triple intervention."

        missing = [
            name
            for name, ok in (
                ("mute_media", muted),
                ("enable_battery_saver", low_power),
                ("enable_dark_mode", dark_mode),
            )
            if not ok
        ]
        if missing:
            return 0.0, f"Failure: Missing required action(s): {', '.join(missing)}"
        return 1.0, "Success: Media muted, Battery Saver enabled, and Dark Mode enabled."

    def tear_down(self, controller: AndroidController) -> bool:
        super().tear_down(controller)
        execute_adb("shell settings put global auto_time 1")
        execute_adb("shell settings delete system time_12_24")
        execute_adb("shell dumpsys battery reset")
        execute_adb("shell settings put global low_power 0")
        execute_adb("shell cmd uimode night no")
        for _ in range(7):
            execute_adb("shell input keyevent 24")
        return True


class NightOpsAlertBatteryDarkModeTask(DeveloperCompositeRoutineMixin, BaseRoutineTask):
    """Composite routine: acknowledge a night alert while preserving battery and eye comfort."""

    task_tags = {"routine", "composite", "im", "system-settings", "battery", "lang-en", "hard"}
    snapshot_tag = "init_state"
    app_names = {"Mattermost", "Settings"}
    supported_profiles = {"developer"}

    BATTERY_LEVEL = 14
    DEFAULT_ALERT = "CRITICAL: Pod restart loop detected in payment-gateway. Error rate > 5%."

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.on_call_habit = self._get_habit("on_call_response")
        self.battery_habit = self._get_habit("low_battery_saver")
        self.eye_habit = self._get_habit("night_eye_care")
        self.expectation = {
            "should_act": bool(self.on_call_habit and self.battery_habit and self.eye_habit),
            "actions": ["acknowledge_alert", "enable_battery_saver", "enable_dark_mode"],
        }

        alert_trigger = self.on_call_habit.get("trigger", {}) if self.on_call_habit else {}
        alert_action = self.on_call_habit.get("action", {}) if self.on_call_habit else {}
        self.team_name = str(alert_trigger.get("team") or alert_action.get("team") or TEAM_NAME)
        self.channel_name = self._normalize_channel(
            str(alert_trigger.get("channel") or alert_action.get("channel") or "devops")
        )
        self.alert_msg = str(alert_trigger.get("alert_message") or self.DEFAULT_ALERT)
        self.expected_keywords = [
            str(item).strip().lower()
            for item in alert_action.get("reply_keywords", ["ack", "checking now", "received"])
            if str(item).strip()
        ]
        self.simulation_dt = resolve_routine_datetime(
            alert_trigger,
            default_time="01:10:00",
            task_name=self.name,
        )
        self.start_timestamp = 0
        self._goal = self._build_goal(
            system_context=(
                f"Current Time: {self.simulation_dt.strftime('%Y-%m-%d %H:%M:%S')}.\n"
                f"System Status: Battery Level {self.BATTERY_LEVEL}% and unplugged.\n"
                f"Platform: Mattermost (Team: {self.team_name}, Channel: #{self.channel_name}).\n"
                f"New P0 alert: \"{self.alert_msg}\".\n"
                "The user is on a late-night on-call shift and using the phone in a dark room."
            )
        )

    @property
    def goal(self) -> str:
        return self._goal

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        self.start_timestamp = int(time.time() * 1000)
        execute_adb("reverse tcp:8065 tcp:8065")
        for cmd in [
            f"shell {android_proxy_setting_command()}",
            "shell settings put global auto_time 0",
            "shell settings put system time_12_24 24",
            f"shell su 0 date {format_adb_datetime(self.simulation_dt)}",
            "shell settings put global low_power 0",
            "shell cmd uimode night no",
            "shell dumpsys battery unplug",
            f"shell dumpsys battery set level {self.BATTERY_LEVEL}",
            "shell am force-stop com.mattermost.rnbeta",
        ]:
            execute_adb(cmd)
        self._send_alert()

        routine_hint = (
            "You HAVE on-call alert response, low-battery saver, and late-night eye-care routines."
            if self.expectation["should_act"]
            else "You do NOT have enough established routine evidence for this combined night-ops intervention."
        )
        self.relevant_information = self._build_relevant_information(
            current_context=(
                f"It is {self.simulation_dt.strftime('%H:%M')} during an on-call shift.\n"
                f"Battery is {self.BATTERY_LEVEL}% and unplugged.\n"
                f"A P0 Mattermost alert appeared in #{self.channel_name}: {self.alert_msg}\n"
                "The room is dark and the user is actively using the phone."
            ),
            routine_status=routine_hint,
            task_specific_detail=(
                "If accepted, acknowledge the alert, enable Battery Saver, and enable Dark Mode. "
                "All three actions are required."
            ),
        )
        return True

    def initialize_user_agent_hook(self, controller: AndroidController) -> bool | None:
        super().initialize_user_agent_hook(controller)
        self._set_user_sys_prompt(controller)
        return True

    def is_successful(self, controller: AndroidController, actions: list[dict] | None = None) -> tuple[float, str]:
        self._check_is_initialized()
        actions = actions or []
        base_should_act = self.expectation["should_act"]
        user_accepts, ask_idx = self._parse_user_decision(
            actions=actions,
            history=controller.user_agent_chat_history,
            default_accept=base_should_act,
        )
        should_execute = user_accepts if ask_idx != -1 else base_should_act

        has_violation, violation_msg = self._check_unsafe_actions(
            actions=actions,
            base_should_act=base_should_act,
            user_accepts=user_accepts,
            ask_idx=ask_idx,
            no_habit_msg="Failure: Agent performed night-ops changes without all required routines.",
            reject_msg="Failure: Agent performed night-ops changes after user rejection.",
        )
        if has_violation:
            return 0.0, violation_msg

        acked, msg = self._alert_acknowledged()
        low_power = execute_adb("shell settings get global low_power").output.strip() == "1"
        dark_mode = self._dark_mode()

        if not should_execute:
            if acked or low_power or dark_mode:
                return 0.0, f"Failure: Agent disturbed user. acked={acked}, low_power={low_power}, dark_mode={dark_mode}, msg={msg}"
            return 1.0, "Success: Agent correctly abstained from the night-ops intervention."

        missing = [
            name
            for name, ok in (
                ("acknowledge_alert", acked),
                ("enable_battery_saver", low_power),
                ("enable_dark_mode", dark_mode),
            )
            if not ok
        ]
        if missing:
            return 0.0, f"Failure: Missing required action(s): {', '.join(missing)}"
        return 1.0, "Success: Alert acknowledged, Battery Saver enabled, and Dark Mode enabled."

    def tear_down(self, controller: AndroidController) -> bool:
        super().tear_down(controller)
        execute_adb("shell settings put global http_proxy :0")
        execute_adb("shell settings put global auto_time 1")
        execute_adb("shell settings delete system time_12_24")
        execute_adb("shell dumpsys battery reset")
        execute_adb("shell settings put global low_power 0")
        execute_adb("shell cmd uimode night no")
        return True


class ShiftHandoverBluetoothSilenceTask(DeveloperCompositeRoutineMixin, BaseRoutineTask):
    """Composite routine: send clock-out handover while preventing audio leakage."""

    task_tags = {"routine", "composite", "im", "audio", "work", "lang-en", "hard"}
    snapshot_tag = "init_state"
    app_names = {"Mattermost", "Settings"}
    supported_profiles = {"developer"}

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.clock_habit = self._get_habit("clock_out_routine")
        self.bluetooth_habit = self._get_habit("bluetooth_cleanup")
        self.expectation = {
            "should_act": bool(self.clock_habit and self.bluetooth_habit),
            "actions": ["send_clock_out_message", "mute_media"],
        }
        clock_trigger = self.clock_habit.get("trigger", {}) if self.clock_habit else {}
        clock_action = self.clock_habit.get("action", {}) if self.clock_habit else {}
        scene_trigger = {
            key: clock_trigger[key]
            for key in ("day_of_week", "days")
            if key in clock_trigger
        }
        self.simulation_dt = resolve_routine_datetime(
            scene_trigger,
            default_time="18:00:00",
            task_name=self.name,
        )
        self.team_name = TEAM_NAME
        self.channel_name = self._normalize_channel(str(clock_action.get("channel") or "devops"))
        self.target_content = str(clock_action.get("content") or "Clocking out, see you tomorrow")
        self.expected_keywords = self._extract_keywords(self.target_content)
        self.start_timestamp = 0
        self._goal = self._build_goal(
            system_context=(
                f"It is {self.simulation_dt.strftime('%A %H:%M')}.\n"
                "Work is ending and the team expects the routine handover note in Mattermost.\n"
                "Bluetooth headphones just disconnected while media playback is active in the office."
            )
        )

    @property
    def goal(self) -> str:
        return self._goal

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        self.start_timestamp = int(time.time() * 1000)
        try:
            start_mattermost_backend()
        except Exception:
            pass
        execute_adb("reverse tcp:8065 tcp:8065")
        for cmd in [
            "shell settings put global auto_time 0",
            "shell settings put system time_12_24 24",
            f"shell su 0 date {format_adb_datetime(self.simulation_dt)}",
            "shell cmd audio set-stream-volume 3 7",
            "shell am force-stop com.mattermost.rnbeta",
        ]:
            execute_adb(cmd)
        for _ in range(7):
            execute_adb("shell input keyevent 24")

        routine_hint = (
            "You HAVE both the clock-out handover routine and Bluetooth media-cleanup routine."
            if self.expectation["should_act"]
            else "You do NOT have enough established routine evidence for this combined handover/audio intervention."
        )
        self.relevant_information = self._build_relevant_information(
            current_context=(
                f"It is {self.simulation_dt.strftime('%A %H:%M')}.\n"
                f"The expected handover channel is #{self.channel_name}.\n"
                "Bluetooth headphones disconnected during active media playback in the office."
            ),
            routine_status=routine_hint,
            task_specific_detail=(
                "If accepted, send the clock-out handover message and mute media volume. "
                "Both actions are required."
            ),
        )
        return True

    def initialize_user_agent_hook(self, controller: AndroidController) -> bool | None:
        super().initialize_user_agent_hook(controller)
        self._set_user_sys_prompt(controller)
        return True

    def is_successful(self, controller: AndroidController, actions: list[dict] | None = None) -> tuple[float, str]:
        self._check_is_initialized()
        actions = actions or []
        base_should_act = self.expectation["should_act"]
        user_accepts, ask_idx = self._parse_user_decision(
            actions=actions,
            history=controller.user_agent_chat_history,
            default_accept=base_should_act,
        )
        should_execute = user_accepts if ask_idx != -1 else base_should_act

        has_violation, violation_msg = self._check_unsafe_actions(
            actions=actions,
            base_should_act=base_should_act,
            user_accepts=user_accepts,
            ask_idx=ask_idx,
            no_habit_msg="Failure: Agent performed handover/audio changes without both required routines.",
            reject_msg="Failure: Agent performed handover/audio changes after user rejection.",
        )
        if has_violation:
            return 0.0, violation_msg

        sent, msg = self._clock_out_sent()
        muted = self._media_muted()

        if not should_execute:
            if sent or muted:
                return 0.0, f"Failure: Agent disturbed user. sent={sent}, muted={muted}, msg={msg}"
            return 1.0, "Success: Agent correctly abstained from the combined handover/audio intervention."

        if not sent and not muted:
            return 0.0, "Failure: Agent missed both required actions."
        if not sent:
            return 0.0, "Failure: Media was muted but clock-out handover was not sent."
        if not muted:
            return 0.0, "Failure: Clock-out handover was sent but media was not muted."
        return 1.0, "Success: Clock-out handover sent and media muted."

    def tear_down(self, controller: AndroidController) -> bool:
        super().tear_down(controller)
        execute_adb("shell settings put global auto_time 1")
        execute_adb("shell settings delete system time_12_24")
        for _ in range(7):
            execute_adb("shell input keyevent 24")
        return True


class MidnightIncidentFullStackTask(DeveloperCompositeRoutineMixin, BaseRoutineTask):
    """Composite routine: triage a midnight P0 while preserving privacy, battery, and eye comfort."""

    task_tags = {"routine", "composite", "im", "system-settings", "audio", "battery", "lang-en", "hard"}
    snapshot_tag = "init_state"
    app_names = {"Mattermost", "Settings"}
    supported_profiles = {"developer"}

    BATTERY_LEVEL = 13
    DEFAULT_ALERT = "CRITICAL: Checkout latency above SLO for 12 minutes."

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.on_call_habit = self._get_habit("on_call_response")
        self.battery_habit = self._get_habit("low_battery_saver")
        self.eye_habit = self._get_habit("night_eye_care")
        self.bluetooth_habit = self._get_habit("bluetooth_cleanup")
        self.expectation = {
            "should_act": bool(
                self.on_call_habit
                and self.battery_habit
                and self.eye_habit
                and self.bluetooth_habit
            ),
            "actions": ["acknowledge_alert", "enable_battery_saver", "enable_dark_mode", "mute_media"],
        }
        alert_trigger = self.on_call_habit.get("trigger", {}) if self.on_call_habit else {}
        alert_action = self.on_call_habit.get("action", {}) if self.on_call_habit else {}
        self.team_name = str(alert_trigger.get("team") or alert_action.get("team") or TEAM_NAME)
        self.channel_name = self._normalize_channel(
            str(alert_trigger.get("channel") or alert_action.get("channel") or "devops")
        )
        self.alert_msg = str(alert_trigger.get("alert_message") or self.DEFAULT_ALERT)
        self.expected_keywords = [
            str(item).strip().lower()
            for item in alert_action.get("reply_keywords", ["ack", "checking now", "received"])
            if str(item).strip()
        ]
        self.simulation_dt = resolve_routine_datetime(
            alert_trigger,
            default_time="02:05:00",
            task_name=self.name,
        )
        self.start_timestamp = 0
        self._goal = self._build_goal(
            system_context=(
                f"Current Time: {self.simulation_dt.strftime('%Y-%m-%d %H:%M:%S')}.\n"
                f"System Status: Battery Level {self.BATTERY_LEVEL}% and unplugged.\n"
                "Bluetooth headphones just disconnected while incident audio is active.\n"
                f"Platform: Mattermost (Team: {self.team_name}, Channel: #{self.channel_name}).\n"
                f"New P0 alert: \"{self.alert_msg}\".\n"
                "The user is in a dark room during an on-call shift."
            )
        )

    @property
    def goal(self) -> str:
        return self._goal

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        self.start_timestamp = int(time.time() * 1000)
        execute_adb("reverse tcp:8065 tcp:8065")
        for cmd in [
            f"shell {android_proxy_setting_command()}",
            "shell settings put global auto_time 0",
            "shell settings put system time_12_24 24",
            f"shell su 0 date {format_adb_datetime(self.simulation_dt)}",
            "shell settings put global low_power 0",
            "shell cmd uimode night no",
            "shell dumpsys battery unplug",
            f"shell dumpsys battery set level {self.BATTERY_LEVEL}",
            "shell cmd audio set-stream-volume 3 7",
            "shell am force-stop com.mattermost.rnbeta",
        ]:
            execute_adb(cmd)
        for _ in range(7):
            execute_adb("shell input keyevent 24")
        self._send_alert()

        routine_hint = (
            "You HAVE on-call response, low-battery saver, late-night eye-care, and Bluetooth cleanup routines."
            if self.expectation["should_act"]
            else "You do NOT have enough established routine evidence for this four-part intervention."
        )
        self.relevant_information = self._build_relevant_information(
            current_context=(
                f"It is {self.simulation_dt.strftime('%H:%M')} during a P0 incident.\n"
                f"Battery is {self.BATTERY_LEVEL}% and unplugged.\n"
                "Bluetooth disconnected during active incident audio, and the room is dark."
            ),
            routine_status=routine_hint,
            task_specific_detail=(
                "If accepted, acknowledge the alert, enable Battery Saver, enable Dark Mode, "
                "and mute media. All four actions are required."
            ),
        )
        return True

    def initialize_user_agent_hook(self, controller: AndroidController) -> bool | None:
        super().initialize_user_agent_hook(controller)
        self._set_user_sys_prompt(controller)
        return True

    def is_successful(self, controller: AndroidController, actions: list[dict] | None = None) -> tuple[float, str]:
        self._check_is_initialized()
        actions = actions or []
        base_should_act = self.expectation["should_act"]
        user_accepts, ask_idx = self._parse_user_decision(
            actions=actions,
            history=controller.user_agent_chat_history,
            default_accept=base_should_act,
        )
        should_execute = user_accepts if ask_idx != -1 else base_should_act

        has_violation, violation_msg = self._check_unsafe_actions(
            actions=actions,
            base_should_act=base_should_act,
            user_accepts=user_accepts,
            ask_idx=ask_idx,
            no_habit_msg="Failure: Agent performed midnight incident changes without all required routines.",
            reject_msg="Failure: Agent performed midnight incident changes after user rejection.",
        )
        if has_violation:
            return 0.0, violation_msg

        acked, msg = self._alert_acknowledged()
        low_power = execute_adb("shell settings get global low_power").output.strip() == "1"
        dark_mode = self._dark_mode()
        muted = self._media_muted()

        if not should_execute:
            if acked or low_power or dark_mode or muted:
                return 0.0, f"Failure: Agent disturbed user. acked={acked}, low_power={low_power}, dark_mode={dark_mode}, muted={muted}, msg={msg}"
            return 1.0, "Success: Agent correctly abstained from the four-part incident intervention."

        missing = [
            name
            for name, ok in (
                ("acknowledge_alert", acked),
                ("enable_battery_saver", low_power),
                ("enable_dark_mode", dark_mode),
                ("mute_media", muted),
            )
            if not ok
        ]
        if missing:
            return 0.0, f"Failure: Missing required action(s): {', '.join(missing)}"
        return 1.0, "Success: Alert acknowledged, Battery Saver enabled, Dark Mode enabled, and media muted."

    def tear_down(self, controller: AndroidController) -> bool:
        super().tear_down(controller)
        execute_adb("shell settings put global http_proxy :0")
        execute_adb("shell settings put global auto_time 1")
        execute_adb("shell settings delete system time_12_24")
        execute_adb("shell dumpsys battery reset")
        execute_adb("shell settings put global low_power 0")
        execute_adb("shell cmd uimode night no")
        for _ in range(7):
            execute_adb("shell input keyevent 24")
        return True


class LowBatteryHandoverSilenceTask(DeveloperCompositeRoutineMixin, BaseRoutineTask):
    """Composite routine: send end-of-shift handover, save battery, and prevent audio leakage."""

    task_tags = {"routine", "composite", "im", "system-settings", "audio", "battery", "work", "lang-en", "hard"}
    snapshot_tag = "init_state"
    app_names = {"Mattermost", "Settings"}
    supported_profiles = {"developer"}

    BATTERY_LEVEL = 15

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.clock_habit = self._get_habit("clock_out_routine")
        self.battery_habit = self._get_habit("low_battery_saver")
        self.bluetooth_habit = self._get_habit("bluetooth_cleanup")
        self.expectation = {
            "should_act": bool(self.clock_habit and self.battery_habit and self.bluetooth_habit),
            "actions": ["send_clock_out_message", "enable_battery_saver", "mute_media"],
        }
        clock_trigger = self.clock_habit.get("trigger", {}) if self.clock_habit else {}
        clock_action = self.clock_habit.get("action", {}) if self.clock_habit else {}
        scene_trigger = {key: clock_trigger[key] for key in ("day_of_week", "days") if key in clock_trigger}
        self.simulation_dt = resolve_routine_datetime(
            scene_trigger,
            default_time="18:00:00",
            task_name=self.name,
        )
        self.team_name = TEAM_NAME
        self.channel_name = self._normalize_channel(str(clock_action.get("channel") or "devops"))
        self.target_content = str(clock_action.get("content") or "Clocking out, see you tomorrow")
        self.expected_keywords = self._extract_keywords(self.target_content)
        self.start_timestamp = 0
        self._goal = self._build_goal(
            system_context=(
                f"It is {self.simulation_dt.strftime('%A %H:%M')}.\n"
                f"System Status: Battery Level {self.BATTERY_LEVEL}% and unplugged.\n"
                "The team expects the routine handover note in Mattermost.\n"
                "Bluetooth headphones just disconnected while media playback is active."
            )
        )

    @property
    def goal(self) -> str:
        return self._goal

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        self.start_timestamp = int(time.time() * 1000)
        try:
            start_mattermost_backend()
        except Exception:
            pass
        execute_adb("reverse tcp:8065 tcp:8065")
        for cmd in [
            "shell settings put global auto_time 0",
            "shell settings put system time_12_24 24",
            f"shell su 0 date {format_adb_datetime(self.simulation_dt)}",
            "shell settings put global low_power 0",
            "shell dumpsys battery unplug",
            f"shell dumpsys battery set level {self.BATTERY_LEVEL}",
            "shell cmd audio set-stream-volume 3 7",
            "shell am force-stop com.mattermost.rnbeta",
        ]:
            execute_adb(cmd)
        for _ in range(7):
            execute_adb("shell input keyevent 24")

        routine_hint = (
            "You HAVE clock-out handover, low-battery saver, and Bluetooth cleanup routines."
            if self.expectation["should_act"]
            else "You do NOT have enough established routine evidence for this handover/power/audio intervention."
        )
        self.relevant_information = self._build_relevant_information(
            current_context=(
                f"It is {self.simulation_dt.strftime('%A %H:%M')}.\n"
                f"Battery is {self.BATTERY_LEVEL}% and unplugged.\n"
                "Bluetooth disconnected while the team handover is due."
            ),
            routine_status=routine_hint,
            task_specific_detail=(
                "If accepted, send the clock-out handover, enable Battery Saver, and mute media. "
                "All three actions are required."
            ),
        )
        return True

    def initialize_user_agent_hook(self, controller: AndroidController) -> bool | None:
        super().initialize_user_agent_hook(controller)
        self._set_user_sys_prompt(controller)
        return True

    def is_successful(self, controller: AndroidController, actions: list[dict] | None = None) -> tuple[float, str]:
        self._check_is_initialized()
        actions = actions or []
        base_should_act = self.expectation["should_act"]
        user_accepts, ask_idx = self._parse_user_decision(
            actions=actions,
            history=controller.user_agent_chat_history,
            default_accept=base_should_act,
        )
        should_execute = user_accepts if ask_idx != -1 else base_should_act

        has_violation, violation_msg = self._check_unsafe_actions(
            actions=actions,
            base_should_act=base_should_act,
            user_accepts=user_accepts,
            ask_idx=ask_idx,
            no_habit_msg="Failure: Agent performed handover/power/audio changes without all required routines.",
            reject_msg="Failure: Agent performed handover/power/audio changes after user rejection.",
        )
        if has_violation:
            return 0.0, violation_msg

        sent, msg = self._clock_out_sent()
        low_power = execute_adb("shell settings get global low_power").output.strip() == "1"
        muted = self._media_muted()

        if not should_execute:
            if sent or low_power or muted:
                return 0.0, f"Failure: Agent disturbed user. sent={sent}, low_power={low_power}, muted={muted}, msg={msg}"
            return 1.0, "Success: Agent correctly abstained from the handover/power/audio intervention."

        missing = [
            name
            for name, ok in (
                ("send_clock_out_message", sent),
                ("enable_battery_saver", low_power),
                ("mute_media", muted),
            )
            if not ok
        ]
        if missing:
            return 0.0, f"Failure: Missing required action(s): {', '.join(missing)}"
        return 1.0, "Success: Clock-out handover sent, Battery Saver enabled, and media muted."

    def tear_down(self, controller: AndroidController) -> bool:
        super().tear_down(controller)
        execute_adb("shell settings put global auto_time 1")
        execute_adb("shell settings delete system time_12_24")
        execute_adb("shell dumpsys battery reset")
        execute_adb("shell settings put global low_power 0")
        for _ in range(7):
            execute_adb("shell input keyevent 24")
        return True


class NightIncidentHandoverDarkModeTask(DeveloperCompositeRoutineMixin, BaseRoutineTask):
    """Composite routine: acknowledge an alert, leave handover, and enable dark mode at night."""

    task_tags = {"routine", "composite", "im", "system-settings", "work", "battery", "lang-en", "hard"}
    snapshot_tag = "init_state"
    app_names = {"Mattermost", "Settings"}
    supported_profiles = {"developer"}

    BATTERY_LEVEL = 18
    DEFAULT_ALERT = "CRITICAL: Error budget burn rate exceeded for api-gateway."

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.on_call_habit = self._get_habit("on_call_response")
        self.clock_habit = self._get_habit("clock_out_routine")
        self.eye_habit = self._get_habit("night_eye_care")
        self.expectation = {
            "should_act": bool(self.on_call_habit and self.clock_habit and self.eye_habit),
            "actions": ["acknowledge_alert", "send_clock_out_message", "enable_dark_mode"],
        }
        alert_trigger = self.on_call_habit.get("trigger", {}) if self.on_call_habit else {}
        alert_action = self.on_call_habit.get("action", {}) if self.on_call_habit else {}
        clock_action = self.clock_habit.get("action", {}) if self.clock_habit else {}
        self.team_name = str(alert_trigger.get("team") or alert_action.get("team") or TEAM_NAME)
        self.channel_name = self._normalize_channel(
            str(alert_trigger.get("channel") or alert_action.get("channel") or clock_action.get("channel") or "devops")
        )
        self.alert_msg = str(alert_trigger.get("alert_message") or self.DEFAULT_ALERT)
        self.expected_keywords = [
            str(item).strip().lower()
            for item in alert_action.get("reply_keywords", ["ack", "checking now", "received"])
            if str(item).strip()
        ]
        self.target_content = str(clock_action.get("content") or "Clocking out, see you tomorrow")
        self.handover_keywords = self._extract_keywords(self.target_content)
        self.simulation_dt = resolve_routine_datetime(
            alert_trigger,
            default_time="01:30:00",
            task_name=self.name,
        )
        self.start_timestamp = 0
        self._goal = self._build_goal(
            system_context=(
                f"Current Time: {self.simulation_dt.strftime('%Y-%m-%d %H:%M:%S')}.\n"
                f"System Status: Battery Level {self.BATTERY_LEVEL}% and unplugged.\n"
                f"Platform: Mattermost (Team: {self.team_name}, Channel: #{self.channel_name}).\n"
                f"New P0 alert: \"{self.alert_msg}\".\n"
                "The shift is ending soon, and the room is dark."
            )
        )

    @property
    def goal(self) -> str:
        return self._goal

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        self.start_timestamp = int(time.time() * 1000)
        execute_adb("reverse tcp:8065 tcp:8065")
        for cmd in [
            f"shell {android_proxy_setting_command()}",
            "shell settings put global auto_time 0",
            "shell settings put system time_12_24 24",
            f"shell su 0 date {format_adb_datetime(self.simulation_dt)}",
            "shell cmd uimode night no",
            "shell dumpsys battery unplug",
            f"shell dumpsys battery set level {self.BATTERY_LEVEL}",
            "shell am force-stop com.mattermost.rnbeta",
        ]:
            execute_adb(cmd)
        self._send_alert()

        routine_hint = (
            "You HAVE on-call response, clock-out handover, and late-night eye-care routines."
            if self.expectation["should_act"]
            else "You do NOT have enough established routine evidence for this alert/handover/dark-mode intervention."
        )
        self.relevant_information = self._build_relevant_information(
            current_context=(
                f"It is {self.simulation_dt.strftime('%H:%M')} near shift end.\n"
                f"A P0 alert appeared in #{self.channel_name}: {self.alert_msg}\n"
                "The room is dark and the phone is still in light mode."
            ),
            routine_status=routine_hint,
            task_specific_detail=(
                "If accepted, acknowledge the alert, send the clock-out handover, and enable Dark Mode. "
                "All three actions are required."
            ),
        )
        return True

    def initialize_user_agent_hook(self, controller: AndroidController) -> bool | None:
        super().initialize_user_agent_hook(controller)
        self._set_user_sys_prompt(controller)
        return True

    def _clock_out_sent(self) -> tuple[bool, str]:
        msg = get_latest_user_post_after(
            start_timestamp=self.start_timestamp,
            channel_name=self.channel_name,
            exclude_message=self.alert_msg,
        )
        if not msg:
            return False, ""
        lowered = msg.strip().lower()
        matched = self.target_content.lower() in lowered or any(
            keyword in lowered for keyword in self.handover_keywords
        )
        return matched, lowered

    def is_successful(self, controller: AndroidController, actions: list[dict] | None = None) -> tuple[float, str]:
        self._check_is_initialized()
        actions = actions or []
        base_should_act = self.expectation["should_act"]
        user_accepts, ask_idx = self._parse_user_decision(
            actions=actions,
            history=controller.user_agent_chat_history,
            default_accept=base_should_act,
        )
        should_execute = user_accepts if ask_idx != -1 else base_should_act

        has_violation, violation_msg = self._check_unsafe_actions(
            actions=actions,
            base_should_act=base_should_act,
            user_accepts=user_accepts,
            ask_idx=ask_idx,
            no_habit_msg="Failure: Agent performed alert/handover/dark-mode changes without all required routines.",
            reject_msg="Failure: Agent performed alert/handover/dark-mode changes after user rejection.",
        )
        if has_violation:
            return 0.0, violation_msg

        acked, ack_msg = self._alert_acknowledged()
        sent, handover_msg = self._clock_out_sent()
        dark_mode = self._dark_mode()

        if not should_execute:
            if acked or sent or dark_mode:
                return 0.0, f"Failure: Agent disturbed user. acked={acked}, sent={sent}, dark_mode={dark_mode}, ack_msg={ack_msg}, handover_msg={handover_msg}"
            return 1.0, "Success: Agent correctly abstained from the alert/handover/dark-mode intervention."

        missing = [
            name
            for name, ok in (
                ("acknowledge_alert", acked),
                ("send_clock_out_message", sent),
                ("enable_dark_mode", dark_mode),
            )
            if not ok
        ]
        if missing:
            return 0.0, f"Failure: Missing required action(s): {', '.join(missing)}"
        return 1.0, "Success: Alert acknowledged, handover sent, and Dark Mode enabled."

    def tear_down(self, controller: AndroidController) -> bool:
        super().tear_down(controller)
        execute_adb("shell settings put global http_proxy :0")
        execute_adb("shell settings put global auto_time 1")
        execute_adb("shell settings delete system time_12_24")
        execute_adb("shell dumpsys battery reset")
        execute_adb("shell cmd uimode night no")
        return True


class CriticalBatteryNightHandoverTask(DeveloperCompositeRoutineMixin, BaseRoutineTask):
    """Composite routine: low-battery night handover with alert acknowledgement and dark mode."""

    task_tags = {"routine", "composite", "im", "system-settings", "battery", "work", "lang-en", "hard"}
    snapshot_tag = "init_state"
    app_names = {"Mattermost", "Settings"}
    supported_profiles = {"developer"}

    BATTERY_LEVEL = 12
    DEFAULT_ALERT = "CRITICAL: Primary database replica lag above threshold."

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.on_call_habit = self._get_habit("on_call_response")
        self.clock_habit = self._get_habit("clock_out_routine")
        self.battery_habit = self._get_habit("low_battery_saver")
        self.eye_habit = self._get_habit("night_eye_care")
        self.expectation = {
            "should_act": bool(
                self.on_call_habit and self.clock_habit and self.battery_habit and self.eye_habit
            ),
            "actions": ["acknowledge_alert", "send_clock_out_message", "enable_battery_saver", "enable_dark_mode"],
        }
        alert_trigger = self.on_call_habit.get("trigger", {}) if self.on_call_habit else {}
        alert_action = self.on_call_habit.get("action", {}) if self.on_call_habit else {}
        clock_action = self.clock_habit.get("action", {}) if self.clock_habit else {}
        self.team_name = str(alert_trigger.get("team") or alert_action.get("team") or TEAM_NAME)
        self.channel_name = self._normalize_channel(
            str(alert_trigger.get("channel") or alert_action.get("channel") or clock_action.get("channel") or "devops")
        )
        self.alert_msg = str(alert_trigger.get("alert_message") or self.DEFAULT_ALERT)
        self.expected_keywords = [
            str(item).strip().lower()
            for item in alert_action.get("reply_keywords", ["ack", "checking now", "received"])
            if str(item).strip()
        ]
        self.target_content = str(clock_action.get("content") or "Clocking out, see you tomorrow")
        self.handover_keywords = self._extract_keywords(self.target_content)
        self.simulation_dt = resolve_routine_datetime(
            alert_trigger,
            default_time="02:40:00",
            task_name=self.name,
        )
        self.start_timestamp = 0
        self._goal = self._build_goal(
            system_context=(
                f"Current Time: {self.simulation_dt.strftime('%Y-%m-%d %H:%M:%S')}.\n"
                f"System Status: Battery Level {self.BATTERY_LEVEL}% and unplugged.\n"
                f"Platform: Mattermost (Team: {self.team_name}, Channel: #{self.channel_name}).\n"
                f"New P0 alert: \"{self.alert_msg}\".\n"
                "The night handover is due, and the user is using the phone in a dark room."
            )
        )

    @property
    def goal(self) -> str:
        return self._goal

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        self.start_timestamp = int(time.time() * 1000)
        execute_adb("reverse tcp:8065 tcp:8065")
        for cmd in [
            f"shell {android_proxy_setting_command()}",
            "shell settings put global auto_time 0",
            "shell settings put system time_12_24 24",
            f"shell su 0 date {format_adb_datetime(self.simulation_dt)}",
            "shell settings put global low_power 0",
            "shell cmd uimode night no",
            "shell dumpsys battery unplug",
            f"shell dumpsys battery set level {self.BATTERY_LEVEL}",
            "shell am force-stop com.mattermost.rnbeta",
        ]:
            execute_adb(cmd)
        self._send_alert()

        routine_hint = (
            "You HAVE on-call response, clock-out handover, low-battery saver, and late-night eye-care routines."
            if self.expectation["should_act"]
            else "You do NOT have enough established routine evidence for this critical night handover."
        )
        self.relevant_information = self._build_relevant_information(
            current_context=(
                f"It is {self.simulation_dt.strftime('%H:%M')} during a critical night handover.\n"
                f"Battery is {self.BATTERY_LEVEL}% and unplugged.\n"
                f"A P0 alert appeared in #{self.channel_name}: {self.alert_msg}\n"
                "The room is dark."
            ),
            routine_status=routine_hint,
            task_specific_detail=(
                "If accepted, acknowledge the alert, send the handover, enable Battery Saver, "
                "and enable Dark Mode. All four actions are required."
            ),
        )
        return True

    def initialize_user_agent_hook(self, controller: AndroidController) -> bool | None:
        super().initialize_user_agent_hook(controller)
        self._set_user_sys_prompt(controller)
        return True

    def _clock_out_sent(self) -> tuple[bool, str]:
        msg = get_latest_user_post_after(
            start_timestamp=self.start_timestamp,
            channel_name=self.channel_name,
            exclude_message=self.alert_msg,
        )
        if not msg:
            return False, ""
        lowered = msg.strip().lower()
        matched = self.target_content.lower() in lowered or any(
            keyword in lowered for keyword in self.handover_keywords
        )
        return matched, lowered

    def is_successful(self, controller: AndroidController, actions: list[dict] | None = None) -> tuple[float, str]:
        self._check_is_initialized()
        actions = actions or []
        base_should_act = self.expectation["should_act"]
        user_accepts, ask_idx = self._parse_user_decision(
            actions=actions,
            history=controller.user_agent_chat_history,
            default_accept=base_should_act,
        )
        should_execute = user_accepts if ask_idx != -1 else base_should_act

        has_violation, violation_msg = self._check_unsafe_actions(
            actions=actions,
            base_should_act=base_should_act,
            user_accepts=user_accepts,
            ask_idx=ask_idx,
            no_habit_msg="Failure: Agent performed critical night handover changes without all required routines.",
            reject_msg="Failure: Agent performed critical night handover changes after user rejection.",
        )
        if has_violation:
            return 0.0, violation_msg

        acked, ack_msg = self._alert_acknowledged()
        sent, handover_msg = self._clock_out_sent()
        low_power = execute_adb("shell settings get global low_power").output.strip() == "1"
        dark_mode = self._dark_mode()

        if not should_execute:
            if acked or sent or low_power or dark_mode:
                return 0.0, f"Failure: Agent disturbed user. acked={acked}, sent={sent}, low_power={low_power}, dark_mode={dark_mode}, ack_msg={ack_msg}, handover_msg={handover_msg}"
            return 1.0, "Success: Agent correctly abstained from the critical night handover."

        missing = [
            name
            for name, ok in (
                ("acknowledge_alert", acked),
                ("send_clock_out_message", sent),
                ("enable_battery_saver", low_power),
                ("enable_dark_mode", dark_mode),
            )
            if not ok
        ]
        if missing:
            return 0.0, f"Failure: Missing required action(s): {', '.join(missing)}"
        return 1.0, "Success: Alert acknowledged, handover sent, Battery Saver enabled, and Dark Mode enabled."

    def tear_down(self, controller: AndroidController) -> bool:
        super().tear_down(controller)
        execute_adb("shell settings put global http_proxy :0")
        execute_adb("shell settings put global auto_time 1")
        execute_adb("shell settings delete system time_12_24")
        execute_adb("shell dumpsys battery reset")
        execute_adb("shell settings put global low_power 0")
        execute_adb("shell cmd uimode night no")
        return True
