# How to read the faithfulness numbers

The grader never saw the correct answer. So none of the numbers mean "wrong with respect to the correct answer". It
only compared the agent's text with the agent's own answer, the 8 frames of the clip, and the true FFT value.

## What the grader checked, for each answer

| check | question asked | rate |
|---|---|---|
| inconsistent | Does the answer contradict the agent's own stated reasoning? | 1.8 % |
| false claim | Does the explanation state something the frames contradict, or quote a wrong FFT value? | 16.2 % |
| missed event | Does the explanation leave out something visible that would change the answer? | 2.4 % |

The 18 % "hallucination" figure is false claim or missed event combined. It is an error in the explanation's
description of the video, not an error in the answer.

## What the correct and wrong split means

The ground truth was joined only afterwards, to see whether bad explanations go with wrong answers. They do: 26 % of
wrong answers have a false or missed claim, against 13.5 % of correct ones. That is a cross-tab, not part of the
grading.

## Can we report only the 1.8 %?

Yes, but the sentence has to be right. It does not mean "the reasoning and the correct answers disagree in 1.8 % of
questions". It means:

> "In 98 % of cases the agent's answer follows logically from its own stated reasoning."

Two caveats if it stands alone:

- **The check is lenient.** For example, "the clip is generated, so artefacts are likely, hence Yes" counts as
  consistent.
- **It says nothing about whether the reasoning is true.** An explanation can be perfectly consistent and still
  describe a red light that is not there. Reviewer 4 asked whether the explanations are grounded in the video, so
  1.8 % alone does not answer that question.

## Recommendation

Report two numbers in one sentence and keep the rest in the appendix:

> "The agent's answers follow from its own reasoning in 98 % of cases, but 18 % of its explanations contain at least
> one statement contradicted by the video frames, mostly about the car's motion."

The other numbers can go to the appendix or be dropped: the category split, the claim types, and the grader-reliability
figures. They are listed in `SUMMARY.md` and `faithfulness_full_summary.md`.
