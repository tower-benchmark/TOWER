# SPDX-License-Identifier: Apache-2.0
"""Private gripper profiles; controller state and dispatch ordering stay explicit."""

import numpy as np


class _AbsoluteGripper:
    def initialize(self, controller, start):
        pass

    def apply(
        self, controller, sim, slot, grip, target, measured, interpolated, applied, modifications
    ):
        # measured_fraction is the historical alias for direct_absolute.
        pass

    def commit(self, controller, applied):
        pass


class _DirectionalGripper(_AbsoluteGripper):
    def desired(self, controller, slot, index, target, modifications):
        return target[index]

    def apply(
        self, controller, sim, slot, grip, target, measured, interpolated, applied, modifications
    ):
        index = grip.index
        desired = self.desired(controller, slot, index, target, modifications)
        desired = self.contact_target(
            controller, sim, slot, index, measured, desired, modifications
        )
        interpolated[index] = desired
        previous = controller._gripper_reference[index]
        delta = desired - previous
        if delta >= 0:
            speed = controller.gripper_open_speed_m_s
            kind = "gripper_generator_open_rate"
        elif previous <= controller.gripper_contact_width_m:
            speed = controller.gripper_contact_close_speed_m_s
            kind = "gripper_generator_contact_close_rate"
        else:
            speed = controller.gripper_close_speed_m_s
            kind = "gripper_generator_close_rate"
        applied[index] = np.clip(
            previous + np.clip(delta, -speed * controller.sim_dt, speed * controller.sim_dt),
            grip.lower,
            grip.upper,
        )
        if applied[index] != desired:
            modifications.append(
                {
                    "index": index,
                    "kind": kind,
                    "before": float(desired),
                    "after": float(applied[index]),
                    "previous_command": float(previous),
                    "max_speed_m_s": speed,
                    "contact_width_m": controller.gripper_contact_width_m,
                }
            )

    def contact_target(self, controller, sim, slot, index, measured, desired, modifications):
        if controller.gripper_contact_hold_force_n is not None:
            if controller._gripper_closed_state[slot]:
                from .contact import ContactClosure

                feedback = sim.finger_feedback(slot)
                if controller.gripper_contact_hold_force_n > feedback["effort_limit_n"]:
                    raise ValueError("Contact hold force exceeds physical finger limit")
                if slot not in controller._contact_closures:
                    controller._contact_closures[slot] = ContactClosure(
                        float(controller._gripper_reference[index]),
                        float(controller.gripper_closed_targets_m[slot]),
                        force_n=controller.gripper_contact_hold_force_n,
                        speed=controller.gripper_contact_close_speed_m_s,
                        free_speed=controller.gripper_close_speed_m_s,
                        contact_width=controller.gripper_contact_width_m,
                    )
                closure = controller._contact_closures[slot]
                desired = closure.update(
                    measured[index],
                    feedback["forces_n"],
                    feedback["velocities_m_s"],
                    finger_positions=feedback["positions_m"],
                )
                modifications.append(
                    dict(
                        index=index,
                        kind="generator_contact_closure",
                        after=float(desired),
                        latched=bool(closure.latched),
                        hold_force_n=controller.gripper_contact_hold_force_n,
                        **feedback,
                    )
                )
            else:
                controller._contact_closures.pop(slot, None)
        return desired

    def commit(self, controller, applied):
        for grip in controller.robot.grippers:
            controller._gripper_reference[grip.index] = applied[grip.index]


class _HysteresisGripper(_DirectionalGripper):
    def initialize(self, controller, start):
        controller._gripper_closed_state = []
        for slot, grip in enumerate(controller.robot.grippers):
            score = (controller.gripper_model_open_widths_m[slot] - start[grip.index]) / (
                controller.gripper_model_open_widths_m[slot]
                - controller.gripper_model_closed_widths_m[slot]
            )
            controller._gripper_closed_state.append(bool(score >= 0.5))

    def desired(self, controller, slot, index, target, modifications):
        score = (controller.gripper_model_open_widths_m[slot] - target[index]) / (
            controller.gripper_model_open_widths_m[slot]
            - controller.gripper_model_closed_widths_m[slot]
        )
        if (
            not controller._gripper_closed_state[slot]
            and score >= controller.gripper_hysteresis_close_score
        ):
            controller._gripper_closed_state[slot] = True
        elif (
            controller._gripper_closed_state[slot]
            and score <= controller.gripper_hysteresis_open_score
        ):
            controller._gripper_closed_state[slot] = False
        desired = (
            controller.gripper_closed_targets_m[slot]
            if controller._gripper_closed_state[slot]
            else controller.gripper_open_targets_m[slot]
        )
        modifications.append(
            {
                "index": index,
                "kind": "gripper_hysteresis_decision",
                "before": float(target[index]),
                "after": float(desired),
                "close_score": float(score),
                "state": "closed" if controller._gripper_closed_state[slot] else "open",
                "close_threshold": controller.gripper_hysteresis_close_score,
                "open_threshold": controller.gripper_hysteresis_open_score,
                "model_closed_width_m": float(controller.gripper_model_closed_widths_m[slot]),
                "model_open_width_m": float(controller.gripper_model_open_widths_m[slot]),
            }
        )
        return desired
