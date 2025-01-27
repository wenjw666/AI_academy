# Keypoint Abstraction using Large Models for Object-Relative Imitation Learning 

Xiaolin Fang ${ }^{* 1}$, Bo-Ruei Huang ${ }^{* 12}$, Jiayuan Mao ${ }^{* 1}$, Jasmine Shone ${ }^{1}$, Joshua B. Tenenbaum ${ }^{1}$, Tomás Lozano-Pérez ${ }^{1}$, Leslie Pack Kaelbling ${ }^{1}$<br>${ }^{1}$ Massachusetts Institute of Technology ${ }^{2}$ National Taiwan University<br>\{xiaolinf,boruei, jiayuanm, jasshone, jbt, tlp, lpk\}@csail.mit.edu


#### Abstract

Generalization to novel object configurations and instances across diverse tasks and environments is a critical challenge in robotics. Keypoint-based representations have been proven effective as a succinct representation for capturing essential object features, and for establishing a reference frame in action prediction, enabling data-efficient learning of robot skills. However, their manual design nature and reliance on additional human labels limit their scalability. In this paper, we propose KALM, a framework that leverages large pre-trained vision-language models (LMs) to automatically generate taskrelevant and cross-instance consistent keypoints. KALM distills robust and consistent keypoints across views and objects by generating proposals using LMs and verifies them against a small set of robot demonstration data. Based on the generated keypoints, we can train keypoint-conditioned policy models that predict actions in keypoint-centric frames, enabling robots to generalize effectively across varying object poses, camera views, and object instances with similar functional shapes. Our method demonstrates strong performance in the real world, adapting to different tasks and environments from only a handful of demonstrations while requiring no additional labels. Website: https://kalm-il.github.io/


## I. Introduction

A long-standing goal in robotics is to develop learning mechanisms that allow efficient acquisition of robot skills across a wide range of tasks, requiring a feasible amount of data while generalizing effectively to different object poses and even instances. Such generalization is crucial for enabling robots to perform robustly in diverse environments. One common strategy for improving data efficiency is to leverage abstractions. Researchers have explored object-centric [1], part-centric [2], and keypoint-centric representations [3] for tasks spanning rigid-body manipulation to deformable object handling, such as ropes and clothes. Among these, keypoints offer a versatile abstraction for many robotic tasks, capturing essential object features with a low-dimensional encoding.

However, despite their potential for data efficiency and generalization, constructing a good keypoint representation can be tedious, both during training and at test time. Such training typically requires human experts to design taskspecific keypoints, while deploying keypoint-centric models in real-world necessitates visual modules capable of detecting these keypoints, which often requires additional data collection and human annotations.

[^0]![](https://cdn.mathpix.com/cropped/2025_01_24_ec5e8a5dce19bcaec800g-1.jpg?height=394&width=868&top_left_y=714&top_left_x=1076)

Fig. 1: Keypoint Abstraction using Large Models for ObjectRelative Imitation Learning (KALM). KALM is a framework that distills keypoint abstraction by prompting and verifying keypoint proposals from large pre-trained models using a small amount of robot demonstration data, which is used to train a keypoint-conditioned policy model. Our method demonstrates strong generalization on multiple real-world manipulation tasks with only 10 demonstrations and no additional labeling effort.

On the other hand, recent advances in large pre-trained models offer a promising alternative for the automatic construction of keypoint-centric representations, but their reliability remains a concern. Previous attempts to use these models for unsupervised keypoint extraction [4] have exclusively relied on the proposals of task-relevant objects or keypoints from pre-trained models and are therefore errorprone. Furthermore, it is unclear how these models, usually trained on little or no robotics data, can be aligned with the constraints and uncertainty in the physical environment. Our key insight is that an appropriate keypoint abstraction should satisfy two essential criteria: task relevance and crossinstance consistency. Task relevance ensures that the keypoints directly support the specific robotic task at hand. Crossinstance consistency ensures that these keypoints are robustly identifiable across multiple instances: from different camera views, over objects having similar functional shapes, in different poses. We hope to use a small amount of robot demonstration data ( 5 to 10 demonstrations) to align the large pre-trained models to the robotics domain.

In this paper, we propose Keypoint Abstraction using Large Models for Object-Relative Imitation Learning (KALM), a framework that distills keypoints by prompting large pretrained models and verifying the proposed keypoints based on a small number of robot demonstration trajectories. For each new task, given a single seeding demonstration video and a short natural language task description, we prompt a
![](https://cdn.mathpix.com/cropped/2025_01_24_ec5e8a5dce19bcaec800g-2.jpg?height=776&width=1701&top_left_y=170&top_left_x=212)

Fig. 2: KALM overview. (a) Keypoint distillation. Given a demonstration video and a task description, we prompt a VLM to generate a coarse-grained region proposal, which is refined into a fine-grained point set via image segmentation models and VLMs. We use a keypoint detection function $\phi$ to identify keypoint correspondences across a handful of demonstration trajectories. The final keypoints set is selected based on correspondence consistency verification. These keypoints are used for training a keypoint-conditioned action model. (b) Inference time. Given a new scene, the keypoint detection function $\phi$ localizes the distilled keypoints. The learned keypoint-conditioned action prediction model generates an object-relative end-effector trajectory based on the keypoint positions and features.
vision-language model to identify candidate object parts that are task-relevant. These candidates are further refined through querying image segmentation models into a candidate set of keypoints. The final keypoint set is selected based on crossinstance consistency computed across available demonstration trajectories of the task. Once the keypoints are identified, we train a diffusion policy model conditioned on these keypoints and their features to generate robot trajectories relative to the object keypoints. Our system demonstrates strong generalization across real-world settings, adapting to changes in environment and object instances.

Overall, our contributions are: First, we propose to distill task-relevant and cross-instance consistent keypoints from large pre-trained models with a combined proposal and verification process. Second, with the extracted keypoints, we build a keypoint-centric, object-relative action representation based on keypoint features and their derived local frames that can be learned by diffusion policy models from a few demonstrations. Our keypoint-conditioned policy model allows robots to generalize their learned behaviors across different environments and object configurations, improving task performance under diverse conditions.

## II. Method

Illustrated in Figure 2, our proposed method distills keypoints through an iterative procedure, alternating between candidate proposals using large pre-trained models and verification based on a small robot demonstration dataset.

Once the keypoints are distilled, we use their visual and geometric features to train a keypoint-conditioned diffusion model for generating object-relative robot actions.

At inference time, our method detects the corresponding points of the previously distilled keypoints in the observed image, predicts the robot actions relative to the detected points, and finally transforms it back into the world frame for execution.

## A. Problem Formulation

Formally, for each skill $\alpha$, we require a single video of the robot successfully executing the task $\mathcal{H}^{\alpha}$, a handful of demonstration trajectories ( 5 to 10 ) $\mathcal{D}^{\alpha}$, and a natural language task description $d^{\alpha}$. Each trajectory $D_{i}^{\alpha}$ contains an initial observation of the scene, represented as a calibrated RGBD image $I_{i}^{\alpha}$ and a robot joint trajectory $\tau_{i}^{\alpha}$. No additional labels such as keypoints are required.

Our first goal is to distill a set of keypoints that are task-relevant and consistent across observed images in all demonstration trajectories $\left\{I_{i}^{\alpha}\right\}$. We represent this keypoint set using 3D locations and their features in the first frame of the demonstration video, denoted as $\mathcal{K}^{\alpha}$. The second goal is to learn a trajectory prediction model, based on the distilled keypoints and demonstration trajectories $\mathcal{D}^{\alpha}$. In this work, we assume the trajectory of each skill is segmented into two distinct phases: an approaching phase, during which the robot moves freely towards the object, and an execution phase. The trajectory prediction model only predicts the execution trajectory, focusing only on the actions needed to manipulate the objects. The segmentation of the trajectory can be done through thresholding the distance to the closest keypoint, or through methods checking more fine-grained trajectory statistics [5], [6], [7]. For brevity, in the following, we omit the superscript $\alpha$ when there are no ambiguities.

## B. Keypoint Desiderata

In our framework, the keypoints serve both as an abstraction of the observational input and the basis of an object-relative action frame. To ensure that the abstraction is effective and the frame is robust to changes in the environment, we define two criteria: task relevance and cross-instance consistency.
Task relevance. To allow generalization to different scene configurations of the same task, the keypoints must be taskrelevant. For example, for the task of lifting the handle of a coffee machine, the points on the handle are ideal candidates whereas those on the water reservoir are not because the latter varies across different machines and does not directly support the task completion.

Given a demonstration $D_{i}=\left\langle I_{i}, \tau_{i}\right\rangle$, the skill description $d$, as well as a keypoint $k$ and the corresponding position $p_{k, i}=\phi\left(k, I_{i}\right)$ in $I_{i}$, a pretrained vision-language model will implicitly assign a score $\psi$ to this keypoint $k$, denoted as $\psi\left(k, D_{i}, d, p_{k, i}\right)$. Our overall goal is to find a set of keypoints $\mathcal{K}$ such that $\psi\left(k, D_{i}, d, p_{k, i}\right)$ is high for all training demonstrations $D_{i}$.
Cross-instance consistency. Furthermore, it is essential that the keypoints are consistently identifiable across observations regardless of the object pose, camera view, or detailed shape of objects. For example, within a task-relevant object part, a point on the corner may be favored over one on a plain surface, due to its saliency and a lower degree of ambiguity.

We evaluate the task relevance of a keypoint by leveraging pre-trained vision-language models, and check the crossinstance consistency using a keypoint detection function $\phi$. Our goal is to find a set of keypoints that are both task-relevant and consistently identifiable.

## C. Keypoint Proposal and Verification

Our keypoint proposal and verification pipeline works in three steps. First, we prompt a pre-trained vision language model (VLM) to select task-relevant image regions. Within the regions, we generate queries to image segmentation models to generate candidate object part masks, which are further ranked and selected by a second query to the VLM. We sample fine-grained keypoint proposals within the selected mask and score them based on consistency across all query images from the training demonstration set. This process will either return a set of final keypoints or declare failure, leading to another iteration of keypoint proposal. The overall process is illustrated in Figure 2a and Algorithm 1.
Coarse-grained region proposal $\boldsymbol{V L M}_{1}(\mathcal{H}, d)$. Our input to the VLM consists of a sequence of images $\mathcal{H}$ showing a single demonstration video of the robot executing the task, along with a natural language description of the skill $d$ (e.g., "open the top drawer"). We aim to identify the regions of interest in the initial image, $I_{0}^{\mathcal{H}}$, associated with this video. We present $I_{0}^{\mathcal{H}}$ with an overlaid grid, where each grid cell is indexed by a unique text label, and query the VLM to select the grid indices corresponding to the task-relevant regions.

In addition to the grid index, we employ zero-shot chain-of-thought [8] prompting, encouraging the VLM to generate

```
Algorithm 1 Keypoint Proposal and Verification Pipeline
Input: Skill description $d$, training set $\mathcal{D}=\left\{\left\langle I_{i}, \tau_{i}\right\rangle\right\}_{i=0}^{N-1}$,
    demonstration video $\mathcal{H}=\left\{I_{t}^{\mathcal{H}}\right\}_{t=0}^{|\mathcal{H}|-1}$
Output: Set of proposed keypoints $\mathcal{K}$ and their matched
    point in the training set $\mathcal{P}:=\phi\left(k, I_{i}\right), \forall k \in \mathcal{K}, i \in I_{i}$
    while True do
        $\mathcal{K} \leftarrow \emptyset ; \mathcal{P} \leftarrow \emptyset$
        $\mathcal{K}_{V L M} \leftarrow \operatorname{VLM}_{1}(H, d) \quad \triangleright$ Region proposal
        $\mathcal{M}_{s} \leftarrow \cup_{k \in \mathcal{K}_{V L M}} S\left(I_{0}^{\mathcal{H}}, k\right) \quad \triangleright$ Image segmentation
        $m \leftarrow V L M_{2}\left(I_{0}^{\mathcal{H}}, k, \mathcal{M}_{S}\right) \quad \triangleright$ Mask selection
        $\mathcal{K}_{F P S} \leftarrow F P S\left(I_{0}^{\mathcal{H}}, m\right) \triangleright$ Fine-grained point sampling
        for each $k^{\prime} \in \mathcal{K}_{F P S}$ do
            if $\frac{1}{N} \sum_{i=0}^{N-1} \mathbb{1}\left[\phi\left(k^{\prime}, I_{i}\right) \neq\right.$ Null $] \geq \mathbf{1}-\delta$ then
                $\mathcal{K}=\mathcal{K} \cup\left\{k^{\prime}\right\} \quad \triangleright$ Add $k^{\prime}$ to the output set
                $\mathcal{P}=\mathcal{P} \cup\left\{\phi\left(k^{\prime}, I_{i}\right) \mid \forall I_{i}\right\}$
        if $\frac{|\mathcal{K}|}{\left|\mathcal{K}_{\text {FPS }}\right|} \geq \gamma$ then
```

a textual description of the target object, and its highlighted part before generating the final prediction.
Mask proposal $\boldsymbol{S}\left(I_{0}^{\mathcal{H}}, k\right)$ and $\boldsymbol{V L M}_{2}\left(I_{0}^{\mathcal{H}}, k, \mathcal{M}_{\boldsymbol{S}}\right)$. Next, for each candidate coarse-grained region (represented as a grid cell in $I_{0}^{\mathcal{H}}$ ), we generate a uniformly distributed set of query points within the cell, and use a point-prompted image segmentation model [9] to generate a set of object-part masks $\mathcal{M}_{S}$ for each of these query points. We apply standard nonmaximum suppression to filter out masks with significant overlaps and discard those with low confidence scores.

Subsequently, we provide the conversation history from the previous VLM query, along with the image $I_{0}^{\mathcal{H}}$ overlaid with all the detected masks $\mathcal{M}_{S}$, as input of a second VLM query $V L M_{2}$. The VLM is tasked with selecting a single mask from $\mathcal{M}_{S}$ that contains the potential task-relevant keypoints. The output of this step is a mask $m$ on the image $I_{0}^{\mathcal{H}}$. By leveraging the VLM's understanding of both the task context and the object-part segmentation masks, we obtain a more detailed representation of the task-relevant region.
Fine-grained Point Sampling $\operatorname{FPS}\left(I_{0}^{\mathcal{H}}, m\right)$. Given the input RGBD image $I_{0}^{\mathcal{H}}$ and the selected mask $m$ from the second VLM query, we apply Farthest Point Sampling [10] to the 3D points located inside the mask $m$ to generate $N_{c}$ candidate points $\mathcal{K}_{F P S}$. We found Farthest Point Sampling to be effective at generating diverse spatially distributed candidate keypoints that are geometrically salient, such as those located on part boundaries. These points are often visually distinct and thus tend to be consistently identifiable. We will then select a set of keypoints from this set through our cross-instance consistency verification.
Keypoint detection function $\phi(k, I)$. Each candidate keypoint $k$ in $\mathcal{K}_{F P S}$ is internally represented by its position $p_{0}^{k}$ and feature vector $F_{r e f}\left(p_{0}^{k}\right)$ in the initial scene $I_{0}^{\mathcal{H}}$. We need to detect their corresponding points in the training set observations $\left\{I_{i}\right\}$. We treat the keypoint detection task as a correspondence matching problem, where the goal is to identify the matching keypoint in a new input image $I$. The keypoint detection function can be written as $\phi(k, I) \rightarrow p$,
where $I \in \mathbb{R}^{H \times W \times 4}$ is a query RGBD image, and $p \in \mathbb{R}^{3}$ is the 3 D position of the keypoint in the query scene. When no matched keypoint is found in $I, \phi$ returns Null.

This is accomplished using a scoring function $\phi^{*}$ based on the per-point feature representation in $I$. Formally, given the keypoint position $p_{0}^{k}$ in $I_{0}^{\mathcal{H}}$, the function $\phi$ returns the position $p$ in $I$ that maximizes the following score:

$$
\begin{equation*}
\phi^{*}\left(p, p_{0}^{k}\right)=\left\langle F_{q}(p), F_{r e f}\left(p_{0}^{k}\right)\right\rangle \tag{1}
\end{equation*}
$$

where $\langle\cdot, \cdot\rangle$ denotes the cosine similarity between two vectors. We also use the same VLM pipeline to obtain a coarse image grid of the target keypoint on the new input image $I$ and discount points outside the mask in this matching process.

To enhance robustness, we implement a local group-based matching strategy that randomly samples $N^{\prime}=8$ neighboring points within a radius of $r=0.02 \mathrm{~m}$ around $p_{0}$, computes the corresponding match for each of them in the input image $I$, and declares a non-match if there is no majority consensus among the sampled points (i.e., if the majority of the matched points in $I$ are not within close proximity of each other).
Cross-instance consistency verification. The previous steps generate a set of proposed keypoints $\mathcal{K}_{F P S}$ from the VLM and their corresponding points (or non-match) in each image of the demonstration trajectories. We evaluate each keypoint based on $\Phi$ that requires a successful correspondence matching in a majority of demonstration trajectories:

$$
\begin{equation*}
\Phi(k):=\frac{1}{N} \sum_{i=0}^{N-1} \mathbb{1}\left[\phi\left(k, I_{i}\right) \neq \mathbf{N u l l}\right] \geq 1-\delta \tag{2}
\end{equation*}
$$

where the acceptance factor $\delta=0.3$ in our experiment.
Our success criteria for the proposal set $\mathcal{K}_{F P S}$ is that the majority of points in this set should be consistent candidates: $\sum_{k} \Phi(k) /\left|\mathcal{K}_{F P S}\right| \geq \gamma$. If there are sufficient consistent candidates in the proposal set $\mathcal{K}_{F P S}$, our algorithm returns those consistent candidates as our final selected keypoints $\mathcal{K}$. Otherwise, we will discard all proposal keypoints and re-prompt VLM to generate another mask and set of proposal points. This step is important for verifying that the VLM selected part is both task-relevant and consistently identifiable.

## D. Learning Keypoint-Centric Trajectory Prediction Models

From the keypoint proposal and verification process, we have determined a set of keypoints $\mathcal{K}$, which captures the most salient and task-relevant object parts for the skill $\alpha$. For each demonstration trajectory $D_{i}$, we also have a set of corresponding points $\mathcal{P}_{i}$ from the previous keypoint detection step. Conditioned on the sparse keypoint locations and features of these detected points, we directly generate a trajectory $\tau_{E E}$ for the 6 DoF pose of the robot's end-effector using the Diffuser [11], a trajectory-level diffusion model. Internally, the Diffuser learns a score function $\epsilon_{\theta}\left(\tau_{E E} \mid \mathbf{C}\right)$ parameterized by $\theta$, which captures the gradient of the data distribution over $\tau_{E E}$, where $\mathbf{C}$ is the conditional input to the diffuser.

We have two key design choices here to facilitate generalization: using a sparse keypoint-based input, and having actions predicted in a keypoint-centric, object-relative coordinate
frame. Specifically, the model only takes the keypoint locations and their features as input, which leverages the keypoint abstraction to obtain invariance to task-irrelevant distractions, such as background and view changes. Meanwhile, we predict the actions relative to the center of these keypoints. The objectrelative nature of the design makes our model invariant to changes in the absolute pose of the camera and the objects.

## E. Implementation

We use GPT-4o [12] as our VLM and Segment-Anything Model [9] (SAM) as our image segmentation model. In the similarity function $\phi^{*}$, we use a combination of pre-trained image features (DINO [13] with FeatUp [14]) and analytic 3D features Fast Point Feature Histograms (FPFH [15]). The overall (cosine) similarity in Equation 1 is defined as:

$$
\begin{align*}
\left\langle F_{q}(p), F_{r e f}\left(p_{0}^{k}\right)\right\rangle & =\lambda_{1} \cdot\left\langle F_{q}^{D I N O}(p), F_{r e f}^{D I N O}\left(p_{0}^{k}\right)\right\rangle  \tag{3}\\
& +\lambda_{2} \cdot\left\langle F_{q}^{F P F H}(p), F_{r e f}^{F P F H}\left(p_{0}^{k}\right)\right\rangle
\end{align*}
$$

where $\lambda_{1}=0.75$ and $\lambda_{2}=0.25$ in our experiments.
For the trajectory prediction model, we employ Diffuser [11]. The trajectory $\tau_{E E}$ is represented as a sequence of $H=48$ poses. For each pose, we use a 10 -dimensional vector that includes a three-dimensional location, 6-dimensional rotation vector [16], and one dimension for the gripper. The input keypoint feature to the Diffuser is DINO and FPFH as in the keypoint detection function. We optimize the model using standard diffusion loss functions. The diffusion model estimates the conditional distribution $p\left(\tau_{E E} \mid \mathbf{C}\right)$, where $\mathbf{C}=\left\{\left\langle p_{k}, F_{k}^{D I N O}, F_{k}^{F P F H}\right\rangle\right\}_{k \in \mathcal{K}}$ represents the set of matched keypoints in $D_{i}, \tau_{E E}$ is the end-effector trajectory, which is used for inference time denoising.

## F. Inference-time Pipeline

At inference time, given a new scene image $I$, we begin by running the keypoint detector $\phi(k, I)$ for all keypoints $k \in \mathcal{K}$, extracting their corresponding position and feature vectors. We then use the learned Diffuser to generate a set of end-effector trajectories. Starting from randomly initialized trajectories sampled from Gaussian noise, the model employs the backward process which iteratively denoises the noisy trajectories through gradient steps guided by the score function $\epsilon_{\theta}$ under given conditions.

Note that the learned trajectory starts relatively close to the target object. We need to ensure reachability and collision-free motion in the environment. Similar to previous work that uses diffusion models as trajectory samplers [17], we use a motion planner (bi-directional RRT [18]) to check whether there is a feasible path to the initial pose of the task trajectory $\tau_{E E}^{t=0}$. If the motion planner returns no valid path, we will test the next predicted trajectory, until all generated trajectories in the set are exhausted, when the algorithm returns failure. Otherwise, we move the end-effector to $\tau_{E E}^{t=0}$ using the approaching path returned by the motion planner, and start $\tau_{E E}$ execution.

## III. EXPERIMENT

In this section, we want to study the following questions:
![](https://cdn.mathpix.com/cropped/2025_01_24_ec5e8a5dce19bcaec800g-5.jpg?height=297&width=885&top_left_y=155&top_left_x=170)

Fig. 3: Testing tasks in Meta-World [19] simulator. We evaluate on 5 tasks in Meta-World with randomized camera and object poses, necessitating the generalization of policies across observational changes. Keypoints are marked in pink for visualization.

- Is the sparse keypoint abstraction sufficient for the conditional diffusion model to predict valid trajectories with only a limited amount of demonstration data?
- Does the keypoint abstraction improve data efficiency compared to the baselines?
- Does the iterative proposal and verification procedure distill appropriate keypoints for the action model learning?
We study the data efficiency of keypoint abstraction in a simulation environment due to the difficulty in collecting a large amount of data for training all baselines in the real world. We also compare different keypoint proposal methods in the real world by measuring the success rate and generalization of the keypoint-conditioned policies.


## A. Simulation Experiments in Meta-World

In this section, we compare our method with other baselines with different input spaces and network architectures using the Meta-World [19] simulator, focusing on the efficacy of different representations in terms of their data efficiency.
Setup. We evaluate on 5 tasks: DrawerOpen, DrawerClose, ButtonSide, Buttontop, and LeverPull, as shown in Figure 3 For each task, we provide an oracle set of keypoints by manually labeling the XML files and computing their 2D or 3D locations in the scene using known object and camera poses. We train and test on scenes with varying camera and object poses. The object poses are uniformly sampled on the table with 2D translations and rotations. The camera viewpoint is sampled around the object with randomized angle and distance with elevation in $\left[0, \frac{\pi}{2}\right]$, azimuth in $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$, and distance in $[1.5,2]$. We exclude random camera angles where the arm obstructs the object in the initial scene.
Baselines. We compare our method against 4 baselines.

- Diffuser [11] (RGB) generates a 6 DoF end-effector trajectory based on an initial RGB observation of the scene and the camera extrinsic, using a 1D convolutional network. We use DINO [13] visual encoder and finetune it at training time to handle the discrepancy between the simulator and real-world images on which DINO is trained.
- 3D Diffuser Actor [5] (RGBD) builds a 3D representation from pre-trained visual features (CLIP [20]) and the point cloud. It generates actions with a diffusion model conditioned on the 3 D representation.
- Diffuser with keypoints: We provide Diffuser with the 2D position of the keypoints as an additional input.
- 3D Diffuser Actor with keypoints: Similarly, we provide the 3D position of the keypoints to the 3D Diffuser Actor.
![](https://cdn.mathpix.com/cropped/2025_01_24_ec5e8a5dce19bcaec800g-5.jpg?height=502&width=852&top_left_y=172&top_left_x=1081)

Fig. 4: Data efficiency. We measure the average success rate across all 5 tasks, with the number of demonstrations increasing from 10 to 500 . Our method, KALM, demonstrates superior data efficiency compared to all baselines.

Few-shot learning results. We present the task success rates of all baselines across 5 tasks trained with only 10 demonstrations in Table I The success rate is averaged over 100 test episodes and the variances are computed across three random seeds. Overall, the Diffuser (RGB) model struggles to predict accurate 3D trajectories. Nonetheless, we observe a significant performance improvement with the keypoints added. The 3D Diffuser Actor, which employs a transformer backbone, requires more training data. It has poor performance in low-data regimes. By contrast, our keypointconditioned diffusion model, taking only sparse inputs of 3D keypoint positions and their visual features, can learn efficiently from very few demonstrations while achieving strong performance.
Data efficiency study. We further evaluate the data efficiency by varying the number of demonstrations from 10 to 500 . We report the average success rate on 5 tasks in Figure 4 (each with three seeds and 100 test episodes). Our method KALM achieves superior data efficiency, reaching peak performance using only 100 demonstrations, whereas the 3D Diffuser Actor requires 500 demonstrations to achieve competitive performance. This validates the effectiveness of sparse keypoint abstraction for trajectory prediction.

## B. Real-World Experiment on Franka Arm

In this section, we investigate whether our proposed iterative procedure generates better keypoints which eventually lead to a higher task success rate for a range of different tasks in the real world. We also explore the generalization capability along different dimensions endowed by using keypoint abstraction such as object poses and object instances. Setup. We carry out real-world experiments on three tasks: 1) Lifting the handle of a coffee machine, 2) Opening the top drawer, and 3) Pouring something into a bowl, as illustrated in Figure 5 We use a Franka Research 3 robot arm with a RealSense D435i RGBD camera mounted on the gripper. For each task, we collect 10 demonstrations and capture the initial image using the gripper-mounted camera.

To evaluate the generalization along different dimensions, we vary the environment in terms of camera and object poses (View), and object instances (Cross obj.). Note that the objects

| \#Demos | Tasks | Without Keypoints |  | With Keypoints |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | $\frac{\text { RGB }}{\text { Diffuser [11] }}$ | $\frac{\text { RGBD }}{3 \mathrm{D} \text { Diffuser Actor [5] }}$ | $\begin{gathered} \hline \text { RGB } \\ \hline \text { Diffuser } \end{gathered}$ | RGBD |  |
|  |  |  |  |  | 3D Diffuser Actor | KALM (Ours) |
| 10 | Draweropen | $30.00 \pm 8.29$ | $30.00 \pm 6.53$ | $62.00 \pm 1.41$ | $29.33 \pm 4.64$ | $77.00 \pm 2.94$ |
|  | DrawerClose | $50.00 \pm 1.41$ | $53.67 \pm 4.19$ | $83.33 \pm 2.62$ | $50.67 \pm 6.60$ | $92.33 \pm 0.47$ |
|  | Buttonside | $32.67 \pm 2.49$ | $37.67 \pm 2.05$ | $49.67 \pm 11.09$ | $38.67 \pm 1.25$ | $79.67 \pm 1.25$ |
|  | Buttontop | $19.00 \pm 3.56$ | $21.00 \pm 4.90$ | $28.00 \pm 8.04$ | $21.00 \pm 4.32$ | $97.33 \pm 0.47$ |
|  | Leverpull | $10.67 \pm 3.30$ | $8.67 \pm 2.36$ | $21.33 \pm 1.70$ | $10.33 \pm 4.19$ | $61.67 \pm 6.13$ |

TABLE I: Few-shot learning in Meta-World. We evaluate our method, KALM, on five manipulation tasks in the Meta-World [19] simulator, using Diffuser [11] and 3D Diffuser Actor [5] as baselines, along with ablation studies on keypoint usage. Our method consistently outperforms these baselines, demonstrating that keypoints serve as an effective abstraction.
![](https://cdn.mathpix.com/cropped/2025_01_24_ec5e8a5dce19bcaec800g-6.jpg?height=337&width=857&top_left_y=704&top_left_x=184)

Fig. 5: Testing tasks in the real world. We evaluate different methods on three tasks in the real world with different objects at different poses, and with different camera angles. The testing assets are illustrated in the figure.

|  | w/o. Verification |  |  | KALM (Ours) |  |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Tasks | View | Cross Obj. |  | View | Cross Obj. |
| Lifting Handle | $1 / 10$ | $0 / 10$ |  | $9 / 10$ | $6 / 10$ |
| Opening Drawer | $2 / 10$ | $0 / 10$ |  | $6 / 10$ | $7 / 10$ |
| Pouring into Bowl | $6 / 10$ | $2 / 10$ |  | $8 / 10$ | $6 / 10$ |

TABLE II: Real-world experiments success rate. We evaluate all models in varying camera and object poses (View), as well as on unseen objects (Cross obj.). Our method demonstrates strong performance under view changes and showcases generalization to objects that are not seen during diffusion training time.
are never seen during the training of the diffusion model, where changes in camera and object poses are also applied. Baseline. To validate the efficacy of the verification procedure, we compare our method against a baseline that directly prompts VLMs to select a set of keypoints without performing cross-instance consistency verification, referred to as "w/o. Verification". Specifically, we make another query to VLM after obtaining the fine-grained point sampled $\mathcal{K}_{F P S}$, and ask it to select $N_{k}=8$ keypoints, as the final output.
Few-shot learning results. We carry out 10 repeated runs for each task. For a fair comparison, we run the baseline and KALM with the same set of 10 setups: the robot is reset to the same initial pose before executing the predicted trajectory, and we reset the objects as close as possible if they were moved by the robot. Table $\Pi$ shows the overall results.

Overall, our method consistently outperforms the baseline, highlighting the importance of cross-instance consistency verification. We observe that the baseline struggles with localizing corresponding keypoints in testing scenes. For example, in the handle-lifting task, the predicted trajectory is
usually centered on the wrong parts of the coffee machine due to keypoint localization failures. Even when the keypoint detection is accurate, the predicted trajectories are significantly worse. We attribute the reason to the poor detection during keypoint distillation at training time, which leads to a lowerquality dataset $\mathcal{P}$ for training the keypoint-conditioned action prediction model, resulting in reduced performance.

With the object-relative action representation, our method achieves strong performance under view changes, abstracting away the absolute position of the camera or object. This enables our method to perform strongly with only 10 demonstrations and no additional labels. We believe this is an efficient way to scale up generalizable skill learning.

## IV. Related Work

Abstractions for action representations Abstractions over raw sensorimotor interfaces have been shown to enhance data-efficient learning and facilitate generalization. Various forms of abstractions have been studied, including contact points [21], [22], [23], [24], objects [25], [26], [27], [28], [29], [30], [1], object parts [31], [32], [33], [34], keypoints [35], [3], [36], [37], and other shape representations [38], [39], [40]. These spatial abstractions can serve as direct inputs to data-driven models [28], [3], [1], or be used to create coordinate frames [41], [34]. Typically, these models represent short-horizon robot behaviors and can be sequentially composed [36], [42] in a fixed order, or integrated into high-level planning algorithms [43]. Our work leverages keypoint-based representations for few-shot imitation learning, and focuses on acquiring such representations automatically without additional data and labels.
Finding keypoint correspondences. Finding functional correspondences between objects [44] has been extensively explored in robotic manipulation, using both analytical methods [45], [46], [47], [48] and data-driven methods [39], [49], [35], [3], [37], [38]. However, the keypoints derived from these approaches are typically tailored to specific tasks, requiring human annotations of task-relevant keypoints at training time. Furthermore, although data-driven methods offer better generalization to novel objects, they require additional data for training, such as labeled keypoints or additional object datasets. In contrast, our method eliminates the need for human-labeled keypoints by leveraging off-theshelf large pre-trained models for vision-language grounding
and visual recognition, enabling automatic discovery of taskrelevant and cross-instance consistent keypoints.
Vision-language models for robotics. A large body of research has focused on utilizing pre-trained language and vision-language models for robotic manipulation, by generating action plans [50], [51], specifying motion constraints [52], [4], and directly predicting robot movements [53], [54]. The primary limitation of these methods is their dependence on primitive sets of low-level controllers or on motions that can be easily described by simple analytical expressions. Additionally, because these approaches rely on one-shot solutions generated by pre-trained models, they are prone to errors produced by these models. Other work proposes generating reward functions [55] or planning models [56] and can learn new policies from data, but they do not tackle the issue of data efficiency.

## V. Conclusion

We propose KALM, a framework that distills task-relevant keypoints by iteratively prompting LMs and verifying consistency using a small amount of data. We use these keypoints as an abstraction to learn a keypoint-conditioned policy model that predicts object-relative robot actions, leveraging the keypoints as observational abstractions and local action frames. Our simulated and real-world experiment shows that our keypoint representation enables data-efficient learning and facilitates generalization to changes in the scene.
Acknowledgements. We gratefully acknowledge support from NSF grant 2214177; from AFOSR grant FA9550-22-1-0249; from ONR MURI grant N00014-22-1-2740; and from ARO grant W911NF-23-1-0034; from MIT Quest for Intelligence; from the MIT-IBM Watson AI Lab; from ONR Science of AI; and from Simons Center for the Social Brain. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of our sponsors.

## References

[1] Y. Zhu, A. Joshi, P. Stone, and Y. Zhu, "Viola: Imitation learning for vision-based manipulation with object proposal priors," in Conference on Robotic Learning, 2023.
[2] W. Liu, J. Mao, J. Hsu, T. Hermans, A. Garg, and J. Wu, "Composable part-based manipulation," in Conference on Robotic Learning, 2023.
[3] W. Gao and R. Tedrake, "kpam 2.0: Feedback control for category-level robotic manipulation," IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 2962-2969, 2021.
[4] W. Huang, C. Wang, Y. Li, R. Zhang, and L. Fei-Fei, "ReKep: SpatioTemporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation," in Conference on Robotic Learning, 2024.
[5] T.-W. Ke, N. Gkanatsios, and K. Fragkiadaki, "3d diffuser actor: Policy diffusion with 3 d scene representations," in Conference on Robotic Learning, 2024.
[6] T. Gervet, Z. Xian, N. Gkanatsios, and K. Fragkiadaki, "Act3d: 3d feature field transformers for multi-task robotic manipulation," in Conference on Robotic Learning, 2023.
[7] M. Shridhar, L. Manuelli, and D. Fox, "Perceiver-actor: A multitask transformer for robotic manipulation," in Conference on Robotic Learning, 2022.
[8] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al., "Chain-of-thought prompting elicits reasoning in large language models," Neural Information Processing Systems, 2022.
[9] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. Berg, W.-Y. Lo, P. Dollar, and R. Girshick, "Segment anything," arXiv:2304.02643, 2023.
[10] Y. Eldar, M. Lindenbaum, M. Porat, and Y. Zeevi, "The farthest point strategy for progressive image sampling," IEEE Transactions on Image Processing, vol. 6, no. 9, pp. 1305-1315, 1997.
[11] M. Janner, Y. Du, J. Tenenbaum, and S. Levine, "Planning with diffusion for flexible behavior synthesis," in International Conference on Machine Learning, 2022.
[12] OpenAI, "Hello GPT-4o," 2024, accessed: 2024-09-15. [Online]. Available: https://openai.com/index/hello-gpt-4o/
[13] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., "Dinov2: Learning robust visual features without supervision," Transactions on Machine Learning Research, 2024.
[14] S. Fu, M. Hamilton, L. E. Brandt, A. Feldmann, Z. Zhang, and W. T. Freeman, "Featup: A model-agnostic framework for features at any resolution," in International Conference on Learning Representations, 2024.
[15] R. B. Rusu, N. Blodow, and M. Beetz, "Fast point feature histograms (FPFH) for 3d registration," in IEEE International Conference on Robotics and Automation, 2009.
[16] Y. Zhou, C. Barnes, L. Jingwan, Y. Jimei, and L. Hao, "On the continuity of rotation representations in neural networks," in IEEE Conference on Computer Vision and Pattern Recognition, 2019.
[17] X. Fang, C. R. Garrett, C. Eppner, T. Lozano-Pérez, L. P. Kaelbling, and D. Fox, "DiMSam: Diffusion Models as Samplers for Task and Motion Planning Under Partial Observability," in IEEE International Conference on Robotics and Automation, 2024.
[18] S. M. LaValle and J. J. Kuffner Jr, "Randomized Kinodynamic Planning," International Journal of Robotics Research, vol. 20, no. 5, pp. 378-400, 2001.
[19] T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine, "Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning," in Conference on Robotic Learning, 2019.
[20] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., "Learning transferable visual models from natural language supervision," in International Conference on Machine Learning, 2021.
[21] J. C. Trinkle and J. J. Hunter, "A Framework for Planning Dexterous Manipulation," in IEEE International Conference on Robotics and Automation, 1991.
[22] X. Ji and J. Xiao, "Planning motions compliant to complex contact states," International Journal of Robotics Research, vol. 20, no. 6, pp. 446-465, 2001.
[23] G. Lee, T. Lozano-Pérez, and L. P. Kaelbling, "Hierarchical Planning for Multi-Contact Non-Prehensile Manipulation," in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2015.
[24] X. Cheng, E. Huang, Y. Hou, and M. T. Mason, "Contact Mode Guided Motion Planning for Quasidynamic Dexterous Manipulation in 3D," in IEEE International Conference on Robotics and Automation, 2022.
[25] C. Diuk, A. Cohen, and M. L. Littman, "An Object-Oriented Representation for Efficient Reinforcement Learning," in International Conference on Machine Learning, 2008.
[26] C. Devin, P. Abbeel, T. Darrell, and S. Levine, "Deep object-centric representations for generalizable robot learning," in IEEE International Conference on Robotics and Automation, 2018.
[27] R. Veerapaneni, J. D. Co-Reyes, M. Chang, M. Janner, C. Finn, J. Wu, J. Tenenbaum, and S. Levine, "Entity Abstraction in Visual ModelBased Reinforcement Learning," in Conference on Robotic Learning, 2020.
[28] C. Wang, R. Wang, A. Mandlekar, L. Fei-Fei, S. Savarese, and D. Xu, "Generalization through hand-eye coordination: An action space for learning spatially-invariant visuomotor control," in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2021.
[29] W. Yuan, C. Paxton, K. Desingh, and D. Fox, "SORNet: Spatial ObjectCentric Representations for Sequential Manipulation," in Conference on Robotic Learning, 2021.
[30] J. Mao, T. Lozano-Pérez, J. Tenenbaum, and L. Kaelbling, "PDSketch: Integrated Domain Programming, Learning, and Planning," in Neural Information Processing Systems, 2022.
[31] J. Aleotti and S. Caselli, "Manipulation Planning of Similar Objects by Part Correspondence," in IEEE International Conference on Robotics and Automation, 2011.
[32] N. Vahrenkamp, L. Westkamp, N. Yamanobe, E. E. Aksoy, and T. Asfour, "Part-based grasp planning for familiar objects," in IEEERAS International Conference on Humanoid Robots, 2016.
[33] A. Myers, C. L. Teo, C. Fermüller, and Y. Aloimonos, "Affordance detection of tool parts from geometric features," in IEEE International Conference on Robotics and Automation, 2015.
[34] W. Liu, J. Mao, J. Hsu, T. Hermans, A. Garg, and J. Wu, "Composable Part-Based Manipulation," in Conference on Robotic Learning, 2023.
[35] L. Manuelli, W. Gao, P. R. Florence, and R. Tedrake, "KPAM: KeyPoint Affordances for Category-Level Robotic Manipulation," in International Symposium of Robotics Research, 2019.
[36] Z. Qin, K. Fang, Y. Zhu, L. Fei-Fei, and S. Savarese, "Keto: Learning keypoint representations for tool manipulation," in IEEE International Conference on Robotics and Automation, 2020.
[37] D. Turpin, L. Wang, S. Tsogkas, S. J. Dickinson, and A. Garg, "GIFT: generalizable interaction-aware functional tool affordances without labels," in Robotics: Science and Systems, 2021.
[38] B. Wen, W. Lian, K. E. Bekris, and S. Schaal, "You Only Demonstrate Once: Category-Level Manipulation from Single Visual Demonstration," in Robotics: Science and Systems, 2022.
[39] A. Simeonov, Y. Du, A. Tagliasacchi, J. B. Tenenbaum, A. Rodriguez, P. Agrawal, and V. Sitzmann, "Neural Descriptor Fields: SE(3)-Equivariant Object Representations for Manipulation," in IEEE International Conference on Robotics and Automation, 2022.
[40] W. Shen, G. Yang, A. Yu, J. Wong, L. P. Kaelbling, and P. Isola, "Distilled feature fields enable few-shot language-guided manipulation," in Conference on Robotic Learning, 2023.
[41] S. Niekum, S. Osentoski, G. Konidaris, and A. G. Barto, "Learning and generalization of complex tasks from unstructured demonstrations," in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012.
[42] Y. Wang, T.-H. Wang, J. Mao, M. Hagenow, and J. Shah, "Grounding, language plans in demonstrations through counterfactual perturbations," in International Conference on Learning Representations, 2024.
[43] C. R. Garrett, R. Chitnis, R. Holladay, B. Kim, T. Silver, L. P. Kaelbling, and T. Lozano-Pérez, "Integrated task and motion planning," Annual review of control, robotics, and autonomous systems, vol. 4, no. 1, pp. 265-293, 2021.
[44] Z. Lai, S. Purushwalkam, and A. Gupta, "The functional correspondence problem," in IEEE International Conference on Computer Vision, 2021.
[45] H. Asada and M. Brady, "The curvature primal sketch," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PAMI8, no. 1, pp. 2-14, 1986.
[46] S. Brandi, O. Kroemer, and J. Peters, "Generalizing pouring actions between objects using warped parameters," in IEEE-RAS International Conference on Humanoid Robots, 2014.
[47] D. Rodriguez and S. Behnke, "Transferring category-based functional grasping skills by latent space non-rigid registration," IEEE Robotics and Automation Letters, vol. 3, no. 3, pp. 2662-2669, 2018.
[48] O. Biza, S. Thompson, K. R. Pagidi, A. Kumar, E. van der Pol, R. Walters, T. Kipf, J. van de Meent, L. L. S. Wong, and R. Platt, "One-shot imitation learning via interaction warping," in Conference on Robotic Learning, 2023.
[49] S. Thompson, L. P. Kaelbling, and T. Lozano-Pérez, "Shape-Based Transfer of Generic Skills," in IEEE International Conference on Robotics and Automation, 2021.
[50] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. Fu, K. Gopalakrishnan, K. Hausman, A. Herzog, D. Ho, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, E. Jang, R. J. Ruano, K. Jeffrey, S. Jesmonth, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, K.-H. Lee, S. Levine, Y. Lu, L. Luu, C. Parada, P. Pastor, J. Quiambao, K. Rao, J. Rettinghouse, D. Reyes, P. Sermanet, N. Sievers, C. Tan, A. Toshev, V. Vanhoucke, F. Xia, T. Xiao, P. Xu, S. Xu, M. Yan, and A. Zeng, "Do as i can and not as i say: Grounding language in robotic affordances," in arXiv:2204.01691, 2022.
[51] J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng, "Code as policies: Language model programs for embodied control," in IEEE International Conference on Robotics and Automation, 2023.
[52] W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, and L. Fei-Fei, "Voxposer: Composable 3d value maps for robotic manipulation with language models," in Conference on Robotic Learning, 2023.
[53] Y. Hu, F. Lin, T. Zhang, L. Yi, and Y. Gao, "Look before you leap: Unveiling the power of gpt-4v in robotic vision-language planning," arXiv:2311.17842, 2023.
[54] K. Fang, F. Liu, P. Abbeel, and S. Levine, "Moka: Open-world robotic manipulation through mark-based visual prompting," Robotics: Science and Systems, 2024.
[55] Y. J. Ma, W. Liang, G. Wang, D.-A. Huang, O. Bastani, D. Jayaraman, Y. Zhu, L. Fan, and A. Anandkumar, "Eureka: Human-level reward design via coding large language models," in International Conference on Learning Representations, 2024.
[56] G. Liu, A. Adhikari, A.-m. Farahmand, and P. Poupart, "Learning Object-Oriented Dynamics for Planning from Text," in International Conference on Learning Representations, 2021.


[^0]:    *: indicates equal contribution.

