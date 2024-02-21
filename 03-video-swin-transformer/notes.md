This is my attempt to reproduce the works of Ze Liu et al. 2021

# # Abstract:
- Pure transformers have attained top accuracy on major video recog benchmarks
- But these models are built on transformer layers that globally connect patches across the temporal dims
- In this paper, they advocate for an 'inductive bias of locality' that leads to better speed/accuracy tradeoff
- Borrowing from the Swin Architecture from Image models, this acheives state of the art accuracy (2021) on Kinetics-400, Kinetics 600 and Something-Something

## Architecture

**input**: 
- Video of size TxHxWx3 consisting of T frames each frame consisting of HxWx3 pixels 
- Each 3D patch of size 2x4x4x3 is treated as a token

- The 3D patch partitioning layer obtains a $\frac{T}{2} \times \frac{H}{4} \times \frac{W}{4}$ 3D tokens
- Each patch/toke consisting of 96 dim features

- A linear embedding layer is then applied to project the features of each token into an arbitrary dimension $C$ 



