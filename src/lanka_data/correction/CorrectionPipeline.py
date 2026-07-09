from lanka_data.correction.CorrectionErrors import (
    CorrectionLoopError,
    DestructiveCorrectionError,
)
from lanka_data.correction.CorrectionPolicy import CorrectionPolicy, RAISE
from lanka_data.correction.rules import DEFAULT_RULES
from lanka_data.correction.WorkingCommand import WorkingCommand

MAX_PASSES = 8


def _apply_rule(wc, rule, policy, corrections):
    new_wc, correction = rule.apply(wc)
    if policy.action(rule.severity) == RAISE:
        raise DestructiveCorrectionError(
            f"{rule.severity} correction refused by policy: "
            + correction.reason,
            correction.from_value,
        )
    corrections.append(correction)
    return new_wc


def _one_pass(wc, rules, policy, corrections):
    applied = False
    for rule in rules:
        if rule.applies(wc):
            wc = _apply_rule(wc, rule, policy, corrections)
            applied = True
    return wc, applied


def _annotate(command, corrections):
    command.corrections = corrections
    if corrections:
        captions = [c.caption for c in corrections]
        command.correction_note = "corrected: " + ", ".join(captions)
    else:
        command.correction_note = None
    return command


def correct(command, policy=None, rules=None):
    policy = policy or CorrectionPolicy()
    rules = DEFAULT_RULES if rules is None else rules
    wc = WorkingCommand.from_command(command)
    corrections = []
    for _ in range(MAX_PASSES):
        wc, applied = _one_pass(wc, rules, policy, corrections)
        if not applied:
            corrected = _annotate(wc.to_command(), corrections)
            return corrected, corrections
    raise CorrectionLoopError(
        "Correction did not reach a fixpoint within" f" {MAX_PASSES} passes",
        command.cmd_id,
    )
