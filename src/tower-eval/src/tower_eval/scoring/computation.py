# SPDX-License-Identifier: Apache-2.0
"""Explicit numerical dependencies; defaults preserve the reference algorithms.

Optimizations implement this interface without changing module globals. The
caller owns their lifetime; scoring rules and detector dispatch stay here.
A replacement must return bit-identical results to these reference functions;
the hook is internal to the benchmark runtime and is not a stable public API.
Imports are deferred because the detectors themselves import this module.
"""


class ReferenceComputation:
    def support_mask(self, recording, target, supporters, dimensions, candidate, rules):
        from .detectors.place import support_mask

        return support_mask(recording, target, supporters, dimensions, candidate, rules)

    def angular_error(self, q, reference):
        from .geometry import angular_error

        return angular_error(q, reference)


REFERENCE = ReferenceComputation()
