# Verifier Agent — Prompt Template
#
# Placeholders (filled by VerifierAgent._fill_template):
#   {task}        — high-level task description
#   {goal}        — desired goal state
#   {observation} — current environment observation (text)
#   {plan}        — proposed plan with numbered steps

## Task
{task}

## Desired Goal
{goal}

## Current State
{observation}

## Proposed Plan
{plan}

---

## Instructions

You are a robotic plan verifier.
Evaluate the proposed plan against the current state and desired goal.

Check the following criteria:

1. **Feasibility**: Can each sub-goal be physically executed given the current state?
2. **Goal Alignment**: If all sub-goals succeed, will the goal be achieved?
3. **Consistency**: Are the sub-goals in a logically correct order?
4. **Completeness**: Are any necessary steps missing?

After your reasoning, respond with a JSON object **only** (no other text):

```json
{
  "feasible": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation of your verdict",
  "corrected_steps": null or ["step1", "step2", ...]
}
```

If the plan is feasible, set `corrected_steps` to null.
If not feasible, provide a corrected ordered list of steps in `corrected_steps`.
