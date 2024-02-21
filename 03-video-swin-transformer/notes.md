This is my attempt to reproduce the works of Ze Liu et al. 2021

Abstract:
- Pure transformers have attained top accuracy on major video recog benchmarks
- But these models are built on transformer layers that globally connect patches across the temporal dims
- In this paper, they advocate for an 'inductive bias of locality' that leads to better speed/accuracy tradeoff
- Borrowing from the Swin Architecture from Image models, this acheives state of the art accuracy (2021) on Kinetics-400, Kinetics 600 and Something-Something
