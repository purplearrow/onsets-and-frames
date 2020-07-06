# Try to use AET to improve Onsets and Frames detection

This is my work in the MIR course. The purpose is to improve Onsets and Frames detection on Maestro and MAPs dataset. Most of the code in this repository is duplicated from [jonghook's repository](https://github.com/jongwook/onsets-and-frames). Thank them for their kindness to share the codes.

The main modifications I made are

1. use librosa to modify keys in the pre-process stage

2. use the idea of AET. Add a small network on frame detection branch to predict key shift. The lost functions are combined to co-train this key-shift network and frame prediction branch.

In short summary, this approach did not improve the p/r/f of Fames detection.

purplearrow
