from __future__ import annotations

from knowu_bench.runtime.controller import AndroidController
from knowu_bench.runtime.utils.helpers import execute_adb
from knowu_bench.runtime.utils.routine_time import format_adb_datetime, resolve_routine_datetime
from knowu_bench.tasks.definitions.routine.base_routine_task import BaseRoutineTask
from knowu_bench.tasks.definitions.routine.composite_hard import DeveloperCompositeRoutineMixin


class DeveloperBatteryDarkModeTask(DeveloperCompositeRoutineMixin, BaseRoutineTask):
    """Medium composite: low battery mitigation plus late-night Dark Mode."""

    task_tags = {"routine", "composite", "system-settings", "battery", "health", "lang-en", "medium"}
    snapshot_tag = "init_state"
    app_names = {"Settings"}
    supported_profiles = {"developer"}

    BATTERY_LEVEL = 14

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.battery_habit = self._get_habit("low_battery_saver")
        self.eye_habit = self._get_habit("night_eye_care")
        self.expectation = {
            "should_act": bool(self.battery_habit and self.eye_habit),
            "actions": ["enable_battery_saver", "enable_dark_mode"],
        }
        trigger = self.eye_habit.get("trigger", {}) if self.eye_habit else {}
        self.simulation_dt = resolve_routine_datetime(
            trigger,
            default_time="01:20:00",
            task_name=self.name,
        )
        self._goal = self._build_goal(
            system_context=(
                f"It is {self.simulation_dt.strftime('%H:%M')}.\n"
                f"System Status: Battery Level {self.BATTERY_LEVEL}% and unplugged.\n"
                "The user is actively using the phone in a dark room."
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

        routine_hint = (
            "You HAVE both the low-battery saver routine and late-night eye-care routine."
            if self.expectation["should_act"]
            else "You do NOT have enough established routine evidence for this battery/display intervention."
        )
        self.relevant_information = self._build_relevant_information(
            current_context=(
                f"It is {self.simulation_dt.strftime('%H:%M')}.\n"
                f"Battery is {self.BATTERY_LEVEL}% and unplugged.\n"
                "The screen is bright while the user is working in a dark room."
            ),
            routine_status=routine_hint,
            task_specific_detail="If accepted, enable Battery Saver and enable Dark Mode. Both actions are required.",
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
            no_habit_msg="Failure: Agent changed battery/display settings without both required routines.",
            reject_msg="Failure: Agent changed battery/display settings after user rejection.",
        )
        if has_violation:
            return 0.0, violation_msg

        low_power = execute_adb("shell settings get global low_power").output.strip() == "1"
        dark_mode = self._dark_mode()

        if not should_execute:
            if low_power or dark_mode:
                return 0.0, f"Failure: Agent disturbed user. low_power={low_power}, dark_mode={dark_mode}"
            return 1.0, "Success: Agent correctly abstained from the battery/display intervention."

        missing = [
            name
            for name, ok in (
                ("enable_battery_saver", low_power),
                ("enable_dark_mode", dark_mode),
            )
            if not ok
        ]
        if missing:
            return 0.0, f"Failure: Missing required action(s): {', '.join(missing)}"
        return 1.0, "Success: Battery Saver and Dark Mode enabled."

    def tear_down(self, controller: AndroidController) -> bool:
        super().tear_down(controller)
        execute_adb("shell settings put global auto_time 1")
        execute_adb("shell settings delete system time_12_24")
        execute_adb("shell dumpsys battery reset")
        execute_adb("shell settings put global low_power 0")
        execute_adb("shell cmd uimode night no")
        return True


class DeveloperBluetoothBatteryTask(DeveloperCompositeRoutineMixin, BaseRoutineTask):
    """Medium composite: Bluetooth media cleanup plus low-battery mitigation."""

    task_tags = {"routine", "composite", "system-settings", "audio", "battery", "lang-en", "medium"}
    snapshot_tag = "init_state"
    app_names = {"Settings"}
    supported_profiles = {"developer"}

    BATTERY_LEVEL = 15

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.bluetooth_habit = self._get_habit("bluetooth_cleanup")
        self.battery_habit = self._get_habit("low_battery_saver")
        self.expectation = {
            "should_act": bool(self.bluetooth_habit and self.battery_habit),
            "actions": ["mute_media", "enable_battery_saver"],
        }
        self._goal = self._build_goal(
            system_context=(
                f"System Status: Battery Level {self.BATTERY_LEVEL}% and unplugged.\n"
                "Bluetooth headphones just disconnected while media playback is active in the office."
            )
        )

    @property
    def goal(self) -> str:
        return self._goal

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        execute_adb("shell settings put system time_12_24 24")
        execute_adb("shell settings put global low_power 0")
        execute_adb("shell dumpsys battery unplug")
        execute_adb(f"shell dumpsys battery set level {self.BATTERY_LEVEL}")
        execute_adb("shell cmd audio set-stream-volume 3 7")
        for _ in range(7):
            execute_adb("shell input keyevent 24")

        routine_hint = (
            "You HAVE both the Bluetooth media-cleanup routine and low-battery saver routine."
            if self.expectation["should_act"]
            else "You do NOT have enough established routine evidence for this audio/battery intervention."
        )
        self.relevant_information = self._build_relevant_information(
            current_context=(
                f"Battery is {self.BATTERY_LEVEL}% and unplugged.\n"
                "Bluetooth headphones disconnected during active media playback in the office."
            ),
            routine_status=routine_hint,
            task_specific_detail="If accepted, mute media and enable Battery Saver. Both actions are required.",
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
            no_habit_msg="Failure: Agent changed audio/battery settings without both required routines.",
            reject_msg="Failure: Agent changed audio/battery settings after user rejection.",
        )
        if has_violation:
            return 0.0, violation_msg

        muted = self._media_muted()
        low_power = execute_adb("shell settings get global low_power").output.strip() == "1"

        if not should_execute:
            if muted or low_power:
                return 0.0, f"Failure: Agent disturbed user. muted={muted}, low_power={low_power}"
            return 1.0, "Success: Agent correctly abstained from the audio/battery intervention."

        missing = [
            name
            for name, ok in (
                ("mute_media", muted),
                ("enable_battery_saver", low_power),
            )
            if not ok
        ]
        if missing:
            return 0.0, f"Failure: Missing required action(s): {', '.join(missing)}"
        return 1.0, "Success: Media muted and Battery Saver enabled."

    def tear_down(self, controller: AndroidController) -> bool:
        super().tear_down(controller)
        execute_adb("shell settings delete system time_12_24")
        execute_adb("shell dumpsys battery reset")
        execute_adb("shell settings put global low_power 0")
        for _ in range(7):
            execute_adb("shell input keyevent 24")
        return True


class DeveloperBluetoothDarkModeTask(DeveloperCompositeRoutineMixin, BaseRoutineTask):
    """Medium composite: Bluetooth media cleanup plus late-night Dark Mode."""

    task_tags = {"routine", "composite", "system-settings", "audio", "health", "lang-en", "medium"}
    snapshot_tag = "init_state"
    app_names = {"Settings"}
    supported_profiles = {"developer"}

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.bluetooth_habit = self._get_habit("bluetooth_cleanup")
        self.eye_habit = self._get_habit("night_eye_care")
        self.expectation = {
            "should_act": bool(self.bluetooth_habit and self.eye_habit),
            "actions": ["mute_media", "enable_dark_mode"],
        }
        trigger = self.eye_habit.get("trigger", {}) if self.eye_habit else {}
        self.simulation_dt = resolve_routine_datetime(
            trigger,
            default_time="01:45:00",
            task_name=self.name,
        )
        self._goal = self._build_goal(
            system_context=(
                f"It is {self.simulation_dt.strftime('%H:%M')}.\n"
                "Bluetooth headphones just disconnected while media playback is active.\n"
                "The user is in a dark, quiet room."
            )
        )

    @property
    def goal(self) -> str:
        return self._goal

    def initialize_task_hook(self, controller: AndroidController) -> bool:
        execute_adb("shell settings put global auto_time 0")
        execute_adb("shell settings put system time_12_24 24")
        execute_adb(f"shell su 0 date {format_adb_datetime(self.simulation_dt)}")
        execute_adb("shell cmd uimode night no")
        execute_adb("shell cmd audio set-stream-volume 3 7")
        for _ in range(7):
            execute_adb("shell input keyevent 24")

        routine_hint = (
            "You HAVE both the Bluetooth media-cleanup routine and late-night eye-care routine."
            if self.expectation["should_act"]
            else "You do NOT have enough established routine evidence for this audio/display intervention."
        )
        self.relevant_information = self._build_relevant_information(
            current_context=(
                f"It is {self.simulation_dt.strftime('%H:%M')} in a quiet dark room.\n"
                "Bluetooth headphones disconnected during active media playback."
            ),
            routine_status=routine_hint,
            task_specific_detail="If accepted, mute media and enable Dark Mode. Both actions are required.",
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
            no_habit_msg="Failure: Agent changed audio/display settings without both required routines.",
            reject_msg="Failure: Agent changed audio/display settings after user rejection.",
        )
        if has_violation:
            return 0.0, violation_msg

        muted = self._media_muted()
        dark_mode = self._dark_mode()

        if not should_execute:
            if muted or dark_mode:
                return 0.0, f"Failure: Agent disturbed user. muted={muted}, dark_mode={dark_mode}"
            return 1.0, "Success: Agent correctly abstained from the audio/display intervention."

        missing = [
            name
            for name, ok in (
                ("mute_media", muted),
                ("enable_dark_mode", dark_mode),
            )
            if not ok
        ]
        if missing:
            return 0.0, f"Failure: Missing required action(s): {', '.join(missing)}"
        return 1.0, "Success: Media muted and Dark Mode enabled."

    def tear_down(self, controller: AndroidController) -> bool:
        super().tear_down(controller)
        execute_adb("shell settings put global auto_time 1")
        execute_adb("shell settings delete system time_12_24")
        execute_adb("shell cmd uimode night no")
        for _ in range(7):
            execute_adb("shell input keyevent 24")
        return True


class DeveloperQuietSystemTripleTask(DeveloperCompositeRoutineMixin, BaseRoutineTask):
    """Medium-hard system-only composite: audio, battery, and display without app navigation."""

    task_tags = {"routine", "composite", "system-settings", "audio", "battery", "health", "lang-en", "medium"}
    snapshot_tag = "init_state"
    app_names = {"Settings"}
    supported_profiles = {"developer"}

    BATTERY_LEVEL = 15

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
            default_time="01:55:00",
            task_name=self.name,
        )
        self._goal = self._build_goal(
            system_context=(
                f"It is {self.simulation_dt.strftime('%H:%M')}.\n"
                f"System Status: Battery Level {self.BATTERY_LEVEL}% and unplugged.\n"
                "Bluetooth headphones just disconnected while media playback is active in a quiet dark room."
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
            "You HAVE Bluetooth cleanup, low-battery saver, and late-night eye-care routines."
            if self.expectation["should_act"]
            else "You do NOT have enough established routine evidence for this three-setting intervention."
        )
        self.relevant_information = self._build_relevant_information(
            current_context=(
                f"It is {self.simulation_dt.strftime('%H:%M')} in a quiet dark room.\n"
                f"Battery is {self.BATTERY_LEVEL}% and unplugged.\n"
                "Bluetooth disconnected during active media playback."
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
            no_habit_msg="Failure: Agent changed settings without all required routines.",
            reject_msg="Failure: Agent changed settings after user rejection.",
        )
        if has_violation:
            return 0.0, violation_msg

        muted = self._media_muted()
        low_power = execute_adb("shell settings get global low_power").output.strip() == "1"
        dark_mode = self._dark_mode()

        if not should_execute:
            if muted or low_power or dark_mode:
                return 0.0, f"Failure: Agent disturbed user. muted={muted}, low_power={low_power}, dark_mode={dark_mode}"
            return 1.0, "Success: Agent correctly abstained from the three-setting intervention."

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
