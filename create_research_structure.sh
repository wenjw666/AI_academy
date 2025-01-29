#!/bin/bash

# 基础目录结构
mkdir -p research_ideas/[idea_name]/{analysis,experiments,resources/{papers,references,figures},development/meeting_notes}

# 创建核心文档
touch research_ideas/[idea_name]/README.md
touch research_ideas/[idea_name]/analysis/{innovation_analysis.md,feasibility_study.md,risk_assessment.md,literature_review.md}
touch research_ideas/[idea_name]/experiments/{experiment_design.md,baseline_comparison.md,results_analysis.md}
touch research_ideas/[idea_name]/development/{technical_details.md,progress_log.md} 