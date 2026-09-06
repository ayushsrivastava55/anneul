# Demo film script

The film is generated, not performed: `video/script/narration.json` is the source of truth for
both the words and the cut, and `video/README.md` explains the pipeline. This file is the
script in a readable form, for anyone who wants to read it rather than watch it.

Seventeen scenes, five minutes forty seconds. Three of them are pitch slides, ten are
screenshots of the product running, and two are title cards.

| # | Scene | On screen |
|---|---|---|
| 1 | Opening | Title card |
| 2 | The problem | Slide |
| 3 | What Anneal does | Slide |
| 4 | Measured, not claimed | Slide |
| 5 | The front door | Landing page |
| 6 | Five questions | New agent form |
| 7 | Tools, in plain words | The tool picker |
| 8 | Your own examples | The examples grid |
| 9 | Your agents | The index, with Run |
| 10 | Step one: Design | Console, design step |
| 11 | Step two: Try it | Console, run step |
| 12 | Step three: Find the faults | Console, diagnose step |
| 13 | Step five: Prove it helped | Console, the gate refusing a change |
| 14 | On a job it had never seen | Slide |
| 15 | How we built it | Slide |
| 16 | The code | GitHub |
| 17 | Close | Title card |

The centrepiece is scene 13. It is a real verdict from `runs/final`: a challenger that scored
0.800 against the incumbent's 0.767 and was thrown away for making three serious mistakes
against one. Every figure spoken in the film is in the committed runs.

Read the exact words in `video/script/narration.json`.
