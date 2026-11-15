"""Opt-in maintainer command replay; importing this module never starts Isaac."""


def create_runner():
    # The maintainer harness supplies checkout/data/build identity and owns admission.
    # This adapter is deliberately unavailable as a public policy runner.
    try:
        from scripts.maintainer.candidate_replay import CandidateRunner
    except ModuleNotFoundError as exc:
        if exc.name not in ("scripts", "scripts.maintainer", "scripts.maintainer.candidate_replay"):
            raise
        raise RuntimeError("maintainer-only; run from a TOWER checkout") from exc

    return CandidateRunner()
