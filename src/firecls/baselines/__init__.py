"""Shared infrastructure for comparing baseline methods against the distilled student.

Every baseline family (CNN classifiers, YOLO detectors, PIDNet segmentation, ...) is reduced
to the same contract: a module that maps a preprocessed image batch to a ``[batch, 4]`` score
tensor ordered as :data:`firecls.baselines.protocol.CLASSES`. The arg-max of those scores is the
predicted image-level label, so every method is scored with the same metric code, on the same
held-out samples, in the same order as the student.
"""
