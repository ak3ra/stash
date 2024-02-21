This is my attempt to reproduce the works of Ze Liu et al. 2021

## Abstract:
- Pure transformers have attained top accuracy on major video recog benchmarks
- But these models are built on transformer layers that globally connect patches across the temporal dims
- In this paper, they advocate for an 'inductive bias of locality' that leads to better speed/accuracy tradeoff
- Borrowing from the Swin Architecture from Image models, this acheives state of the art accuracy (2021) on Kinetics-400, Kinetics 600 and Something-Something

## Architecture


-  Input is Video of size $T \times H \times W \times 3$ consisting of $T$ frames each frame consisting of $H \times W \times 3$ pixels 
- Each 3D patch of size $2 \times 4 \times 4 \times 3$ is treated as a token

- The 3D patch partitioning layer obtains a $\frac{T}{2} \times \frac{H}{4} \times \frac{W}{4}$ 3D tokens
- Each patch/toke consisting of 96 dim features

- A linear embedding layer is then applied to project the features of each token into an arbitrary dimension $C$ 

### Video Swin Transformer Block

- The multi-head self attention layer is replaced with the 3D shifted window based multi-head self attention module
- Then followed by a feed forward 2 layer MLP with GELU non linearity between
- Layer Normalization is performed before each MSA module and Feed Forward Network
- A residual connection is applied after each module 


### 3D shifted Window based MSA Module 
As said earlier, they introduce a locality inductive bias to the self attention module which is effective in video recognition

#### MSA on non overlapping windows
- 01 - Given a video of $T^{'} \times H^{'} \times W^{'} $ 3D tokens and the size of $P \times M \times M$, the windows are arranged to evenly partition the video input in a non overlapping manner 
- 02 - MSA is performed within each non overlapping 3D window 


#### 3D shifted Windows
- Because MSA is applied on each non overlapping 3D window, there lacks connection across different windows thus limiting representation power of the architecture
- To fix this, they perform cross window connections (2D Swin transformer mechanisms) into 3D.
- This is best explained visually, but they essentially move the window partitions across temporal height and width axes and eventually result into more windoes
- At the end of trhe day, the shifterd config is followed such that the final number of windows for computation is still 8
- Similarly to 2D Swin Transformers, this introduces connections betweeen neighboring non-overlapping 3D windows in consecutive layers
- As a result of this, SOTA is achieved on Kinetics and Something-Something-V2

#### 3D Relative Bias 
- Previous work said this is important, therefore they also do it $B \in \mathbb{R}^{p^{2} \times {M^2} \times M^2}$ for each head as 

$$ 
Attention (QKV) = Softmax(\frac{QK^T}{\sqrt{d} + B})V
$$

## Variants

They release 4 different Variants 
- Swin-T: $C$ = 96, layer numbers $= {2,2,6,2}$
- Swin-S: $C$ = 96, layer numbers $= {2,2,18,2}$
- Swin-B: $C$ = 128, layer numbers $= {2,2,18,2}$
- Swin-L: $C$ = 192, layer numbers $= {2,2,18,2}$

Where $C$ denotes the channel number of the hidden layer in the first stage


## Experiments
### Datasets 

**Human Action Recognition** 
- Kinetics-400
- Kinetics-600 

**Temporal Modelling** 
- Something-Something-V2



