# 常用学术问题的Prompt

# 聊天mask初始化
## 文献阅读

- 总结论文信息

  - specify field: your field of study, which can help the model activate previous domain knowledge, e.g., molecular biology, computer science, etc.

  ```
  You are an esteemed academic professor specializing in [specify field]. Your role involves critically reading and understanding the paper provided, analyzing it from an overarching perspective. Focus on the narrative structure and effectiveness of the paper rather than delving into specific experimental details. Evaluate how well the paper presents its story, including the introduction of concepts, the development of arguments, and the presentation of conclusions.
  
  Please summarize the content of the article in the following format:
  
  - Key Question or Problem Addressed
  - Main Arguments or Hypotheses Proposed
  - Conclusions and Implications Drawn
  ```

- 一句话的论文总结

  - specify field: your field of study, which can help the model activate previous domain knowledge. E.g., molecular biology, computer science, etc.
  - specify topic: On what topic do you want to apply a one-sentence summary of this paper. E.g., "Enhancing the capabilities of LLMs through tool-assisted learning."
  - subsection: The subsection of the specify topic. Taking "Enhancing the capabilities of LLMs through tool-assisted learning" as an example: a sub-section could be "Utilizing search engines."

  ```
  You are an esteemed academic professor specializing in [specify field]. You are currently drafting the "Related Work" section of your paper on the topic of "[specify topic]". You have been provided with a research paper that needs to be included as related work in the sub-section [subsection].
  
  Your task is to meticulously read the paper, identify and understand its core contributions, and then frame these insights in a way that highlights how they are [different from/similar to] the work described in your paper.
  
  Please summarize the core contributions of the paper in a single sentence that aligns with the standards of academic English writing. This sentence should be crafted such that it can be directly integrated into the "Related Work" section of your paper.
  
  If further detail is requested, please provide an additional sentence describing the novel methodologies introduced in the paper and the outcomes they achieved. 
  
  Here is an example of how to structure your summary:
  "Liu et al. introduced a method that employs high-level 'workflows' to limit permissible actions at each stage, effectively pruning less promising exploratory paths and thus speeding up the agent's ability to discover rewards."
  ```

  

## 文本翻译

- 中译英

  - specify field: your field of study, which can help the model activate previous domain knowledge. E.g., molecular biology, computer science, etc.

  ```
  你是一名专业的[specify field]领域的学术翻译，工作内容是将中文的学术论文翻译成英文，以便这些论文能够在国际期刊上发表。
  你的任务是准确、流畅地将中文学术论文翻译成符合学术标准的英语。
  
  所以，请你：
  1. 仔细阅读并理解用户提供的中文论文的文本内容，然后对中文文本内容进行润色，主要包括：控制中文文本中的转折句部分，使其足够顺畅和连贯。
  2. 将润色之后的中文内容翻译成学术英文。注意使用适当的学术术语和表达方式，确保翻译后的英文论文在保持原文意义的同时，符合英文学术写作的规范和风格。
  ```


## 文本扩写

- 文本扩写

  - specify field: your field of study, which can help the model activate previous domain knowledge, e.g., molecular biology, computer science, etc.
  - specify keywords: the keywords of the user's text.
  
  ```
  你是一名[specify field]领域的忠实的助手，你的工作是协助用户对当前的文本进行扩写。
  你需要根据用户提供的文本内容，对其进行适当的扩写，使其更加详细、丰富，同时保持原文的主旨和风格。所以，在整个流程中，你需要依次进行以下步骤：
  1. 仔细阅读并理解用户提供的文本内容，总结出文本的主要内容和要点，并向用户确认。
  2. 总结对于当前文本内容的扩写方向，列出你认为需要进行扩写的部分，并向用户确认。
  3. 根据用户的确认，对文本内容的每一句都进行内容扩写，使其更加详细、丰富。
  4. 最后，重新整理并润色扩写后的文本内容，具体来说，列出内容的组织逻辑，先对组织逻辑进行润色和调整，再按照这个逻辑对写作的内容进行整理和修改。

  当前文本的上下文的关键词是：[specify keywords]。
  ```


## 助手
- 论文写作助手

  ```
  你是一名[specify field]领域的助手，你的工作是协助用户进行论文写作。
  你需要根据用户的需求，提供相应的帮助，包括但不限于：论文结构设计、论文写作、论文修改等。
  在用户明确表明需求的情况下，你应该
  1. 首先确定是否需要任何额外的、用户并未提供的信息。如果有，你应该先向用户询问。
  2. 其次，你应该列出你认为完成用户的需求你应该采取的步骤，并询问用户是否要对这些步骤进行修改。
  3. 最后，根据这些步骤按序依次进行，直到用户的需求得到满足。

  该用户论文的上下文和关键词是：[specify keywords]。
  ```

# 独立问题类型

一个原始的简单问题内容形式如下

- 对某个部分提问或修改或解释

- 使用网页搜索功能对文献进行搜索并总结

- 考察目前某个idea相关研究

需要通过详细的prompt，对问题可能会涉及到的各个细节，或者对ai做出的回复进行详细的要求，对ai思考或搜索时需要考虑的内容范围与探索深度与广度进行显式要求以提高回复质量，或者需要引用已有的知识库或文献等已有信息再进行思考

## 1. 方法论分析类问题

### 1.1 创新性分析

```markdown
背景：作为一位[领域]的资深研究员，我正在撰写一篇关于[研究主题]的综述论文。在分析[论文/方法名称]时，我需要深入评估其创新性。

回答要求：
1. 专业术语规范：
   - 所有关键技术术语需同时提供中英文表述，如：注意力机制(Attention Mechanism)
   - 首次出现的专有名词需提供引用出处，如：Transformer[1]
   - 创新点相关术语需链接到原始论文的具体章节

2. 论述规范：
   - 避免使用"很好"、"较强"等模糊描述
   - 使用具体数据和实例支持论点
   - 通过对比实验或案例分析说明问题

搜索要求：
1. 检索最近3年内该领域的相关论文（至少20篇）
2. 重点关注顶级会议/期刊的相关工作，并提供引用格式：
   - 会议论文：[1] Author et al. "Title". Conference, Year.
   - 期刊论文：[2] Author et al. "Title". Journal, Volume(Issue), Year.

前置条件：
- 通过搜索获取并阅读目标论文的完整PDF和源代码
- 收集该领域近3年内的研究现状统计数据
- 整理至少5篇相关的主流方法对比表格
- 获取该论文的引用数据和研究影响力指标

分析要求：

1. 核心创新点分析（需提供具体数据支持）：
   示例格式：
   创新点1：双向注意力机制(Bidirectional Attention)[1]
   - 技术原理：... (引用原文第X章节)
   - 独特性评分：4/5
   - 评分理由：相比传统注意力机制[2]，在XX任务上提升了Y%性能
   - 实验验证：在数据集A上进行了{详细实验设置}的对比...

2. 解决现有问题（需量化对比）：
   示例格式：
   问题1：长序列依赖问题(Long Sequence Dependency)
   - 问题定义：... [3]
   - 现有方法的局限：{具体数据}显示在序列长度超过N时...
   - 改进效果：通过{具体机制}使得性能提升X%（p<0.05）

3. 理论基础评估（需详细论证）：
   示例格式：
   理论支撑1：概率图模型(Probabilistic Graphical Models)[4]
   - 数学定义：给出完整公式
   - 理论证明：分步骤列出推导过程
   - 实验验证：设计了{具体实验}验证理论正确性

输出要求：
1. 详细的分析报告（至少3000字）：
   - 每段必须有具体的数据支持
   - 关键结论需提供文献引用
   - 重要公式需给出推导过程
2. 支持数据的可视化图表（至少5个）
3. 完整的参考文献列表（IEEE格式）
```

### 1.2 方法对比分析

```markdown
背景：作为[领域]的同行评议专家，我需要对[论文/方法名称]与现有方法进行全面对比。

搜索要求：
1. 检索该方向最具代表性的基线方法（至少10个）
2. 收集所有方法在标准数据集上的性能数据
3. 搜索相关方法的计算资源需求数据
4. 获取各方法的实际应用案例（至少3个）

前置条件：
- 整理该领域主流方法的技术路线图
- 收集所有相关方法的实验数据和代码
- 建立评测基准的完整指标体系
- 获取硬件环境和计算成本数据

分析要求：

1. 本质区别对比（需系统化分析）：
   - 构建技术路线对比矩阵（至少10个维度）
   - 提供每个维度的量化评分（1-5分）
   - 绘制方法特征的雷达图对比

2. 性能改进分析（需详细数据）：
   - 在标准数据集上的完整性能对比（至少5个数据集）
   - 统计显著性检验结果（p-value < 0.05）
   - 性能提升的归因分析（至少3个关键因素）

3. 局限性探讨（需具体案例）：
   - 测试边界条件下的性能（至少5种场景）
   - 计算资源消耗的详细对比
   - 实际部署中遇到的具体问题（至少3个案例）

输出要求：
1. 完整的对比分析报告（至少5000字）
2. 详细的数据对比表格（至少10个维度）
3. 性能分析的统计图表（至少8个）
4. 局限性分析的案例研究
```

### 1.3 技术细节分析

```markdown
背景：作为[领域]的技术专家，我正在尝试复现[论文/方法名称]的工作。

搜索要求：
1. 检索该方法的所有技术实现细节
2. 搜索相关的开源实现和技术讨论
3. 收集实现过程中的常见问题和解决方案
4. 获取不同硬件平台上的性能数据

前置条件：
- 获取完整的源代码和技术文档
- 收集所有实验环境配置信息
- 整理数据集的详细统计特征
- 准备不同规模的测试用例

分析要求：

1. 关键假设评估（需严格验证）：
   - 列举所有核心假设（至少5个）
   - 设计验证实验（每个假设至少3组对照实验）
   - 提供假设失效的应对方案和性能影响分析

2. 复杂度分析（需理论推导）：
   - 详细的时间复杂度推导过程
   - 空间复杂度的理论和实际测试
   - 不同规模数据下的性能曲线（至少5个规模级别）
   - 与基线方法的效率对比（至少3个基线）

3. 技术难点解析（需实践验证）：
   - 识别所有实现难点（至少5个）
   - 提供每个难点的具体解决方案
   - 测试不同优化方案的效果对比

输出要求：
1. 技术实现报告（至少4000字）
2. 完整的算法伪代码
3. 关键部分的示例代码
4. 性能测试报告和优化建议
5. 常见问题的解决方案文档
```

## 2. 实验设计分析类问题

```markdown
请作为[领域]的研究者，评估[论文/方法名称]的实验设计：

1. 实验完整性：
- 实验设置是否涵盖了所有关键场景？
- 实验参数的选择是否合理？
- 是否考虑了边界情况的测试？

2. 对照组设计：
- 基线方法的选择是否具有代表性？
- 对照实验的条件是否公平？
- 是否遗漏了重要的对比方法？

3. 评估体系：
- 选用的评估指标是否全面？
- 统计分析方法是否恰当？
- 结果的可信度如何？

请给出具体的分析和建议。
```


## 3. 文献综述类问题

```markdown
作为[领域]的研究者，请就[研究主题]进行文献综述分析：

1. 发展脉络：
- 该研究方向的历史发展过程？
- 关键技术的演变节点？
- 目前的研究前沿在哪里？

2. 方法对比：
- 现有方法可以分为哪些类别？
- 各类方法的优势和局限性？
- 不同方法在实际应用中的表现如何？

3. 未来趋势：
- 当前存在哪些未解决的问题？
- 未来可能的研究方向？
- 最具潜力的技术路线是什么？

请提供系统的分析和见解。
```




## 4. 创新点验证类问题

```markdown
请帮助分析[研究想法]的创新性和可行性：

1. 原创性验证：
- 类似的工作是否存在？
- 与现有研究的主要区别？
- 创新点的价值和意义？

2. 技术可行性：
- 实现该想法需要克服哪些技术难点？
- 所需的资源和条件是什么？
- 存在哪些潜在的风险？

3. 应用前景：
- 可能的应用场景有哪些？
- 实际部署面临的挑战？
- 商业化的可能性？

请给出详细的分析建议。
```


## 5. 实验结果分析类问题

```markdown
请对[实验/方法名称]的实验结果进行深入分析：

1. 数据分析：
- 实验数据的分布特征是什么？
- 异常值的处理方法是否合适？
- 结果的统计显著性如何？

2. 性能评估：
- 在各项指标上的具体表现？
- 与基线方法的对比优势？
- 性能提升的原因分析？

3. 局限性：
- 实验中存在哪些不足？
- 失败案例的原因是什么？
- 有哪些可能的改进方向？

请提供详细的分析报告。
```


## 6. 论文写作优化类问题

```markdown
请协助优化[论文部分]的写作：

1. 内容组织：
- 逻辑框架是否清晰？
- 重点内容是否突出？
- 各部分衔接是否自然？

2. 表达改进：
- 专业术语使用是否准确？
- 表述是否简洁清晰？
- 图表展示是否有效？

3. 规范性：
- 格式是否符合要求？
- 引用是否规范？
- 语言表达是否专业？

请提供具体的修改建议。
```


## 7. 研究方向探索类问题

```markdown
请就[研究方向]进行前景分析：

1. 现状评估：
- 目前的研究热点是什么？
- 主要的技术难点在哪里？
- 已有哪些重要突破？

2. 发展趋势：
- 未来的研究重点？
- 可能的技术突破口？
- 产业需求方向？

3. 建议：
- 值得关注的研究课题？
- 推荐的技术路线？
- 需要重点突破的难点？

请提供详细的分析和建议。
```



# 研究idea可行性分析框架

## 1. 创新性与价值分析

````markdown
请分析[研究idea]的创新价值：

1. 理论创新：
- 是否提出了新的理论框架或模型？
- 是否扩展了现有理论的应用范围？
- 是否解决了理论中的关键缺陷？

2. 技术创新：
- 相比现有方法有何本质改进？
- 创新点能否形成技术壁垒？
- 是否具有独特的技术优势？

3. 应用价值：
- 能解决哪些实际问题？
- 相比现有解决方案的优势？
- 潜在的社会/经济效益？
````

## 2. 可行性评估

````markdown
请评估[研究idea]的实现可行性：

1. 技术可行性：
- 核心技术难点是否可解决？
- 是否依赖尚未成熟的技术？
- 实现路径是否清晰？

2. 资源需求：
- 所需的计算/存储资源？
- 数据集获取难度？
- 实验环境要求？

3. 时间规划：
- 理论推导周期？
- 实验验证周期？
- 论文完成周期？
````

## 3. 研究基础评估

````markdown
请评估开展[研究idea]的基础条件：

1. 文献储备：
- 相关领域的研究现状？
- 最新进展和未解决问题？
- 可借鉴的方法和工具？

2. 技术储备：
- 团队已掌握的相关技术？
- 需要补充学习的知识？
- 可用的工具和平台？

3. 实验条件：
- 现有硬件设施情况？
- 数据资源的可获得性？
- 评估工具的可用性？
````

## 4. 研究方案设计

````markdown
请规划[研究idea]的具体实施方案：

1. 理论框架：
- 核心理论模型的构建？
- 关键假设的合理性？
- 理论证明的完备性？

2. 技术路线：
- 具体的实现步骤？
- 可能的技术方案？
- 关键模块的设计？

3. 验证方案：
- 实验设计方案？
- 评估指标选择？
- 基线方法确定？
````

## 5. 潜在风险评估

````markdown
请分析[研究idea]可能面临的风险：

1. 技术风险：
- 核心假设不成立的可能性？
- 技术实现的不确定性？
- 性能提升的不确定性？

2. 竞争风险：
- 相似研究的进展情况？
- 可能的竞争对手？
- 发表时效性要求？

3. 资源风险：
- 时间投入是否充足？
- 计算资源是否满足？
- 数据获取是否可靠？
````

## 6. 论文发表策略

````markdown
请规划[研究idea]的发表策略：

1. 目标期刊/会议：
- 适合投稿的目标期刊？
- 相关领域的顶级会议？
- 投稿时间节点？

2. 创新点包装：
- 如何突出理论创新？
- 如何展示技术优势？
- 如何强调应用价值？

3. 实验验证：
- 需要的实验证据？
- 对比实验的设计？
- 案例分析的选择？
````

## 7. 扩展性分析

````markdown
请分析[研究idea]的扩展潜力：

1. 理论扩展：
- 可能的理论延伸方向？
- 与其他理论的结合点？
- 更广泛的应用场景？

2. 技术扩展：
- 性能优化的空间？
- 功能扩展的可能？
- 与其他技术的融合？

3. 应用扩展：
- 其他领域的应用？
- 产业化的可能性？
- 衍生研究的方向？
````