# Goal: roomdesk

Someone asks for a meeting room in one message. Book the smallest room that fits everyone and is free for the whole slot, using the tools. If nothing fits, book nothing and say so.

## What done means
Leave the system in exactly the state the request asks for. Then report what the end state is as a single JSON object under the key `state`.

## Rules
- Every fact in your answer must come from the task text or from a tool result. Never guess.
- Your final message is the answer itself: nothing before it, nothing after it, no
  explanation and no code fence.

- Work in the order: reset -> list_rooms -> check_availability -> book_room -> confirmed
- Avoid book_rooming a smaller room just to have a room booked in case nothing is available
- Ensure only one room is booked, not more, to maintain accuracy and avoid unnecessary bookings
- Do not assume availability based on past states, recheck availability if a room seems available
- Never book a larger room if a smaller one can fit everyone, as the goal is to use the smallest room
- Do not assume the existence of rooms or availability outside the given slots, use tools to verify

## Tools
- `reset`: Restore the calendar to the start of the day. Called by the evaluator per task.
- `list_rooms`: List every meeting room with how many people it seats and whether it has a screen.
- `check_availability`: Say whether one room is free for a whole slot. Hours are 24h, end is exclusive.
- `book_room`: Book a room for a slot. Fails if the room is too small or the slot is already taken.
- `confirmed`: Every booking made so far today. The evaluator reads this, the agent does not need it.

See tools.yaml for the exact argument schemas.
