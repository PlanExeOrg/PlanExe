"""
Provides a system for generating initial time estimates for hierarchical tasks.

This module defines a `Node` class to represent tasks in a tree-like plan.
The core functionality lies in the `Node.resolve_duration()` method, which
recursively traverses the task tree to distribute and calculate time durations
based on a set of predefined hierarchical rules.

Key behaviors of the duration resolution process:
- **Child Resolution First:** Children's durations are resolved recursively
  before the parent's duration is finalized.
- **Rounding:** All calculated durations (for both parents and children)
  are rounded *up* to the nearest whole number (integer `Decimal`) during
  the `resolve_duration` process. This ensures compatibility with systems
  requiring integer durations (like Gantt charts) and prevents fractional values.
- **Initial None Handling:** Tasks initially set to `None` duration, which
  end up as leaves (no children or children resolve to 0), will finalize at `Decimal(0)`.
  After rounding, this remains `Decimal(0)`. `apply_minimum_duration` will
  then turn this into `Decimal(1)`.
- **Bottom-Up Summation:** If a parent task's duration was initially `None`,
  its resolved duration becomes the sum of its children's *resolved and rounded*
  durations. The parent's final duration is also rounded.
- **Top-Down Distribution (if parent > sum of children):**
    - If a parent has a specified duration (`is not None`) and its *rounded* value
      is greater than the sum of its children's *resolved and rounded* durations:
        - If *all* children originally had a specified duration (none were `None`),
          the parent's total *rounded* duration is distributed *evenly* among
          all children. The result of this division is *rounded up* before
          assigning to children, overwriting their initial/resolved values.
          The parent's duration becomes the sum of these newly assigned *rounded*
          child durations.
        - If *some* children originally had `None` duration, the excess duration
          (`rounded parent duration - sum of already resolved & rounded children durations`)
          is distributed among *only* those children that were originally `None`.
          The distributed amount per child is *rounded up* before assignment.
          Children with initial specified durations keep their resolved and rounded
          values unless the override rule above applied (which it wouldn't in this case).
          The parent's duration becomes the sum of all final *rounded* child durations.
- **Top-Down Constraint (if parent <= sum of children):**
    - If a parent has a specified duration (`is not None`) and its *rounded* value is
      less than or equal to the sum of its children's *resolved and rounded*
      durations, children initially set to `None` will receive `Decimal(0)` duration
      (which is rounded to `Decimal(0)`). Children with initial specified durations
      keep their resolved and rounded values. The parent's final duration will be
      the sum of its children's resolved and rounded durations (which will be
      >= the original rounded parent duration).
- **Preventing Negatives:** Initial durations must be non-negative or `None`.
  Calculated durations during distribution are clamped at `Decimal(0)` *before* rounding.
- **Final Consistency:** The parent's duration is *always* set to the final
  sum of its children's durations *after* the children have been resolved,
  rounded, and any distribution/assignment logic for them has occurred. This
  final parent duration is also rounded. All durations are guaranteed to be
  `Decimal` instances representing whole numbers (integers >= 0) after
  `resolve_duration`.

All durations are handled internally using Python's `Decimal` type for precision
during calculation, but rounded to integers for final storage.
The `Node` class also provides a `to_dict()` method for serializing the task tree,
which converts the integer `Decimal` durations to integer types.

The primary purpose of this module is to serve as an initial estimator,
providing a complete set of baseline durations for all tasks in a plan before
potentially more refined estimation techniques (e.g., human review, LLM-based
adjustments) are applied.
"""
from typing import Dict, Optional
from decimal import ROUND_CEILING, Decimal as D

class Node:
    def __init__(self, id: str, duration: Optional[D] = None):
        if not isinstance(id, str):
            raise ValueError("id must be a string")
        if duration is not None:
            if not isinstance(duration, D):
                 raise ValueError("duration must be a Decimal or None")
            if duration < D(0):
                 raise ValueError("duration cannot be negative")

        self.id = id
        self.duration = duration
        self.children = []
        # Flag to remember if the duration was initially None, needed for distribution logic
        self._had_none_duration = duration is None

    def add_child(self, child: 'Node') -> 'Node':
        """Adds a child node and returns it."""
        if not isinstance(child, Node):
            raise TypeError("Can only add Node instances as children")
        self.children.append(child)
        return child

    def to_dict(self):
        """Convert the node and its children to a JSON-compatible dictionary.
        The duration is converted from Decimal to integer."""
        result = {
            "id": self.id,
            # Duration is already rounded to an integer Decimal by resolve_duration
            "duration": int(self.duration) if self.duration is not None else None,
        }
        if self.children: # Use if self.children to avoid empty list in output for leaves
            result["children"] = [child.to_dict() for child in self.children]
        return result

    def _round_duration(self, duration: D) -> D:
        """Rounds a Decimal duration up to the nearest integer Decimal (whole number)."""
        if duration is None:
            return None # Should not happen with current logic flow, but defensive
        # Ensure non-negative before rounding. rounding works fine with D(0)
        duration = max(D(0), duration)
        # Use quantize for explicit rounding to a whole number place (D(1) is '1')
        return duration.quantize(D(1), rounding=ROUND_CEILING)

    def resolve_duration(self):
        """
        Recursively resolves durations in the subtree rooted at this node.
        Applies hierarchical rules: bottom-up summation and top-down distribution.
        Ensures parent duration is the sum of children's resolved durations at the end.
        Rounds all final durations to the nearest integer (ceiled).
        """
        # 1. Recursively resolve children first (bottom-up pass)
        for child in self.children:
            child.resolve_duration()

        # Round child durations immediately after recursive call returns
        for child in self.children:
            if child.duration is not None: # Should always be true after child.resolve_duration()
                 child.duration = self._round_duration(child.duration)

        # If no children, this is a leaf node.
        if not self.children:
             # If a leaf node had None duration, set it to 0
             if self.duration is None:
                 self.duration = D(0)
             # Round leaf duration as well
             if self.duration is not None:
                 self.duration = self._round_duration(self.duration)
             return

        # 2. At this point, all children have resolved durations (Decimal), and are rounded.
        # Calculate the sum of the children's resolved durations.
        sum_children_duration = sum(child.duration for child in self.children) # Sum of rounded decimals

        # Store parent's *initial* state before potential modification
        initial_parent_duration = self.duration # This could be None or a Decimal
        parent_was_initially_none = self._had_none_duration

        # 3. Apply parent logic based on initial state and children's sum

        if parent_was_initially_none:
            # Case A: Parent had no initial duration, its duration is the sum of children.
            # Children's durations are already resolved and rounded.
            pass # Parent duration will be set to sum_children_duration in step 4

        elif initial_parent_duration is not None:
            # Case B: Parent had an initial duration.
            # Round initial parent duration for comparison/distribution calculation
            rounded_initial_parent_duration = self._round_duration(initial_parent_duration)

            # Count children that were initially None
            unassigned_children_count = sum(1 for child in self.children if child._had_none_duration)

            # Compare rounded parent duration with sum of rounded children durations
            if rounded_initial_parent_duration > sum_children_duration:
                 # Parent has more duration than the sum of its children's resolved durations.
                 # This excess duration needs to be distributed.

                 if unassigned_children_count == 0:
                    # Case B1: Parent > sum, AND all children originally had durations.
                    # Apply the override rule: distribute parent's total rounded duration evenly among ALL children.
                    duration_per_child = rounded_initial_parent_duration / D(len(self.children))
                    for child in self.children:
                         # Round the distributed amount before assigning
                         child.duration = self._round_duration(duration_per_child)

                 else:
                    # Case B2: Parent > sum, AND some children originally had None duration.
                    # Distribute the *remaining* duration (rounded_parent_duration - sum of already assigned children)
                    # among the children that were initially None.
                    # already_assigned_sum is the sum of children whose original duration was NOT None.
                    # Their durations are already rounded at the start of step 2.
                    already_assigned_sum = sum(child.duration for child in self.children if not child._had_none_duration)
                    remaining_duration_to_distribute = rounded_initial_parent_duration - already_assigned_sum

                    # Ensure remaining duration is non-negative before distributing
                    remaining_duration_to_distribute = max(D(0), remaining_duration_to_distribute)

                    duration_per_unassigned_child = D(0)
                    if unassigned_children_count > 0:
                         duration_per_unassigned_child = remaining_duration_to_distribute / D(unassigned_children_count)

                    for child in self.children:
                         if child._had_none_duration:
                            # Round the distributed amount before assigning
                            # Ensure non-negative before rounding
                            child.duration = self._round_duration(max(D(0), duration_per_unassigned_child))

            # else (rounded_initial_parent_duration <= sum_children_duration)
            # Case B3: Parent duration is less than or equal to the sum of children's resolved durations.
            # The parent's duration doesn't constrain the children in a top-down way.
            # Children originally None or with initial durations kept their (rounded) values.
            # No distribution from parent needed.

        # 4. Final Step: Ensure parent's duration is the sum of its children's *final* durations.
        # This provides the invariant: parent_duration == sum(children_durations).
        # Since children durations are already rounded, their sum will likely be an integer.
        # Round the final parent duration to be absolutely sure it's an integer Decimal.
        final_sum_children = sum(child.duration for child in self.children)
        self.duration = self._round_duration(final_sum_children)

    def apply_minimum_duration(self):
        """
        In real life, no piece of work takes less than 1 unit of work.
        In the draft plan usually have several tasks with a duration of 0.
        Thus this function ensures that no task has a duration less than 1.

        For leaf nodes (no children), sets duration to 1 if it was 0.
        For parent nodes, checks children and updates parent sum.
        Assumes resolve_duration has already been called and durations are integer Decimals >= 0.
        Maintains the invariant that parent duration equals sum of children's durations.
        """
        # First recursively apply minimum duration to all children
        for child in self.children:
            child.apply_minimum_duration()

        # For parent nodes, ensure each child has at least duration 1
        # and parent duration is sum of children
        # Leaf node minimum handled below
        if self.children:
             for child in self.children:
                 # Child duration is already integer Decimal >= 0 after resolve_duration
                 if child.duration < D(1):
                    child.duration = D(1) # Set to integer Decimal 1

             # Update parent duration to be sum of children (sum of integer Decimals is integer)
             self.duration = sum(child.duration for child in self.children)

        # If this is a leaf node (no children)
        else: # not self.children
            # Leaf duration is already integer Decimal >= 0 after resolve_duration
            # Set minimum duration to 1 if it was 0
            if self.duration < D(1):
                self.duration = D(1) # Set to integer Decimal 1

    def task_id_to_duration_dict(self) -> Dict[str, D]:
        """
        Returns a dictionary of task IDs and their durations.
        These durations are Decimal instances representing whole numbers (integers >= 0)
        after resolve_duration and apply_minimum_duration have been called.
        """
        result = {self.id: self.duration}
        for child in self.children:
            result.update(child.task_id_to_duration_dict())
        return result
