from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


NOTEBOOK_PATH = Path("Garbage_Net.ipynb")


def clean(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(clean(text))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(clean(text))


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
            "mimetype": "text/x-python",
            "file_extension": ".py",
        },
    }

    cells: list[nbf.NotebookNode] = []

    cells.append(md(
        """
        # Garbage Net:一种基于手工特征与传统机器学习方法的垃圾图像分类模型

        ## 摘要

        本项目关注的是 10 类垃圾图像分类问题。我采用的路线不是直接训练深度神经网络，而是先把每张图像转换为颜色、纹理、HOG、形状、GIST、SIFT/ORB BoF 等多组手工特征，再通过传统机器学习模型完成分类。整个实验过程围绕一个问题展开：在类别不平衡、类内差异较大、部分类别视觉边界模糊的情况下，怎样让手工特征和传统模型尽可能稳定地发挥作用。

        为了避免只展示最终结果，我把建模拆成几个阶段：先用固定参数做单特征基线，再对有潜力的单特征模型做轻量调参；随后根据前两步结果设计特征融合方案，并在 Logistic Regression、KNN、Linear SVM、Decision Tree、Random Forest、XGBoost 和 LightGBM 上做广筛；在确认 LightGBM 是最强单模型后，继续比较特征消融和类别不平衡处理；最后用 stacking 集成学习多个小模型与强 LightGBM anchor 之间的互补信息。

        本文把最终选出的最优 stacking 集成模型命名为 **Garbage Net**。Garbage Net 由 6 个 base learner 和一个 Logistic Regression meta learner 组成，在最终 test 集上取得 accuracy = 0.8238、macro-F1 = 0.8097、weighted-F1 = 0.8228。这里的 test 集只在模型封存后使用一次，因此最终结果可以作为独立泛化表现来看待。
        """
    ))

    cells.append(md(
        """
        ## 1. 实验设计与整体流程

        我把整个实验设计成一条由浅到深的建模链路，而不是一开始就堆复杂模型。这样做的好处是，每一步都能回答一个具体问题：哪些特征本身有效、哪些特征互补、哪些模型更适合高维混合手工特征、类别权重是否真的提升少数类，以及 stacking 是否能带来稳定增益。

        完整流程如下：

        ```text
        数据审查与固定划分
          -> 图像预处理
          -> F0-F6 手工特征提取
          -> 固定参数单特征基线
          -> 单特征候选轻量调参
          -> 自适应特征融合与 7 类模型广筛
          -> LightGBM / XGBoost 强模型精调
          -> LightGBM 特征消融与类别不平衡处理
          -> stacking 集成
          -> 最终 test 集评价与错误分析
        ```

        评价指标上，我把 macro-F1 作为主指标，因为它不会让样本多的类别完全主导结论；同时保留 accuracy、weighted-F1 和少数类 recall 均值，用来观察模型在整体准确性和少数类召回之间的取舍。
        """
    ))

    cells.append(code(
        """
        from pathlib import Path
        import os
        import json
        import warnings

        PROJECT_ROOT = Path.cwd()
        CACHE_ROOT = PROJECT_ROOT / ".notebook_cache"
        (CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
        (CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

        import contextlib
        import io
        import numpy as np
        import pandas as pd
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt
            import seaborn as sns
        from PIL import Image as PILImage
        from IPython.display import display, Markdown, SVG

        warnings.filterwarnings("ignore", category=UserWarning)
        pd.set_option("display.max_columns", 80)
        pd.set_option("display.max_colwidth", 80)
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        sns.set_theme(style="whitegrid", context="notebook")

        LABEL_ORDER = [
            "battery", "biological", "cardboard", "clothes", "glass",
            "metal", "paper", "plastic", "shoes", "trash"
        ]

        REPORT_DIR = Path("data/processed/model_reports")
        FIGURE_DIR = Path("data/processed/figures")

        def read_csv(path: str | Path) -> pd.DataFrame:
            return pd.read_csv(Path(path))

        def read_json(path: str | Path):
            with open(Path(path), "r", encoding="utf-8") as f:
                return json.load(f)

        def show_table(df: pd.DataFrame, columns=None, n=None, sort_by=None, ascending=False):
            out = df.copy()
            if sort_by is not None:
                out = out.sort_values(sort_by, ascending=ascending)
            if columns is not None:
                out = out[columns]
            if n is not None:
                out = out.head(n)
            display(out.reset_index(drop=True))
            return out

        def short_text(value, max_len=70):
            text = str(value)
            return text if len(text) <= max_len else text[:max_len - 3] + "..."

        def bar_labels(ax, fmt="{:.3f}", rotation=0):
            for container in ax.containers:
                ax.bar_label(container, fmt=fmt, padding=3, fontsize=9, rotation=rotation)

        print("Notebook helpers ready.")
        """
    ))

    cells.append(md(
        """
        ## 2. 数据集介绍与固定划分

        数据集包含 10 个类别：battery、biological、cardboard、clothes、glass、metal、paper、plastic、shoes 和 trash。各类别样本量并不均衡，其中 clothes、glass、plastic 等类别较多，trash 最少。这个特点直接影响后续评价方式：如果只看 accuracy，模型可能更偏向样本多的类别，因此我把 macro-F1 作为主要指标，并额外观察 trash、battery、biological 这几个少数类的 recall 均值。

        数据划分固定为 train / validation / test = 70% / 15% / 15%。train 用来拟合特征处理步骤和模型，validation 用来做模型筛选、特征组合比较和调参选择，test 在最终模型确定后才使用一次。
        """
    ))

    cells.append(code(
        """
        class_dist = read_csv("data/processed/metadata/class_distribution.csv")
        split_summary = read_csv("data/processed/splits/split_summary.csv")

        split_table = split_summary.rename(columns={
            "label": "class",
            "total": "total",
            "train_count": "train",
            "validation_count": "validation",
            "test_count": "test",
        })[["class", "total", "train", "validation", "test"]]

        display(Markdown("**类别与固定划分统计**"))
        display(split_table)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

        order = class_dist.sort_values("count", ascending=False)["label"]
        sns.barplot(data=class_dist, x="label", y="count", order=order, ax=axes[0], color="#4C78A8")
        axes[0].set_title("Class Distribution")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Images")
        axes[0].tick_params(axis="x", rotation=35)

        split_plot = split_table.set_index("class").loc[LABEL_ORDER, ["train", "validation", "test"]]
        split_plot.plot(kind="bar", stacked=True, ax=axes[1], color=["#4C78A8", "#F58518", "#54A24B"])
        axes[1].set_title("Stratified Train / Validation / Test Split")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Images")
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].legend(title="split")

        plt.tight_layout()
        plt.show()

        totals = {
            "total_images": int(class_dist["count"].sum()),
            "train": int(split_summary["train_count"].sum()),
            "validation": int(split_summary["validation_count"].sum()),
            "test": int(split_summary["test_count"].sum()),
        }
        display(pd.DataFrame([totals]))
        """
    ))

    cells.append(md(
        """
        ## 3. 图像预处理与特征体系

        正式建模使用的是已经统一到 256×256 的图像版本。这样可以减少原始尺寸差异对传统特征提取的影响，也方便把每张图像稳定地转换成定长向量。预处理阶段会根据不同特征需要生成 RGB、HSV、灰度图、二值/前景近似等表示。

        我把特征分成 F0-F6 七组。F0 是像素基线，只作为参照；F1-F6 是正式使用的手工特征。全量手工特征组合 C11 包含 Color、Texture、Shape、GIST、HOG 和 BoF，总维度为 3165。后续所有涉及 scaler、视觉词典、模型参数选择的步骤都只在 train 或 train 内部划分上拟合，避免把 validation/test 信息提前泄露到训练过程中。
        """
    ))

    cells.append(code(
        """
        feature_report = read_csv("data/processed/feature_reports/feature_extraction_report.csv")

        main_features = feature_report[
            feature_report["feature_id"].isin([
                "F0_pixels_gray", "F0_pixels_rgb", "F1_color", "F2_texture",
                "F3_hog", "F4_shape", "F5_gist", "F6_bof"
            ])
        ].copy()
        main_features["role"] = main_features["planned_role"].map(lambda x: short_text(x, 48))

        display(Markdown("**特征组与维度**"))
        display(main_features[[
            "feature_id", "role", "actual_dim", "train_shape",
            "validation_shape", "test_shape", "nan_count_total", "inf_count_total", "status"
        ]].reset_index(drop=True))

        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        sns.barplot(
            data=main_features.sort_values("actual_dim", ascending=False),
            x="feature_id", y="actual_dim", ax=ax, color="#4C78A8"
        )
        ax.set_title("Feature Dimensions")
        ax.set_xlabel("")
        ax.set_ylabel("Dimension")
        ax.tick_params(axis="x", rotation=25)
        bar_labels(ax, fmt="{:.0f}")
        plt.tight_layout()
        plt.show()

        preview_path = Path("data/processed/previews/preprocessing_preview_grid.jpg")
        if preview_path.exists():
            display(Markdown("**预处理抽样检查图**"))
            display(PILImage.open(preview_path))
        """
    ))

    cells.append(md(
        """
        ## 4. 阶段二：固定参数单特征基线

        这一阶段我先不做复杂调参，而是用固定参数把每一组特征单独送入基础模型。这样可以把问题拆开：如果某个特征单独就没有任何有效信息，后面融合时就要谨慎；如果某个特征 macro-F1 一般但少数类 recall 好，它仍然可能在融合模型里提供互补价值。

        固定参数模型包括 Logistic Regression、KNN、Linear SVM 和 Decision Tree。这个阶段的重点不是最终性能，而是形成对特征强弱的第一轮判断。
        """
    ))

    cells.append(code(
        """
        s2_fixed = read_csv(REPORT_DIR / "stage2_single_feature_baselines.csv")
        s2_fixed = s2_fixed[s2_fixed["status"].eq("ok")].copy()

        top_fixed = s2_fixed.sort_values("validation_macro_f1", ascending=False).head(10)
        display(Markdown("**固定参数单特征 baseline Top 10**"))
        display(top_fixed[[
            "feature_id", "model", "feature_dim", "validation_accuracy",
            "validation_macro_f1", "minority_recall_mean",
            "recall_trash", "recall_battery", "recall_biological"
        ]].reset_index(drop=True))

        best_fixed_idx = s2_fixed.groupby("feature_id")["validation_macro_f1"].idxmax()
        best_fixed = s2_fixed.loc[best_fixed_idx].sort_values("validation_macro_f1", ascending=False)
        display(Markdown("**每组特征的最佳固定参数结果**"))
        display(best_fixed[[
            "feature_id", "model", "feature_dim",
            "validation_macro_f1", "minority_recall_mean"
        ]].reset_index(drop=True))

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        sns.barplot(
            data=best_fixed,
            x="feature_id", y="validation_macro_f1",
            ax=axes[0], color="#4C78A8"
        )
        axes[0].set_title("Best Fixed Baseline by Feature")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Validation Macro-F1")
        axes[0].tick_params(axis="x", rotation=25)
        axes[0].set_ylim(0, max(0.65, best_fixed["validation_macro_f1"].max() + 0.05))
        bar_labels(axes[0])

        sns.scatterplot(
            data=s2_fixed,
            x="validation_macro_f1", y="minority_recall_mean",
            hue="feature_id", style="model", s=90, ax=axes[1]
        )
        axes[1].set_title("Macro-F1 vs Minority Recall")
        axes[1].set_xlabel("Validation Macro-F1")
        axes[1].set_ylabel("Minority Recall Mean")
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.show()
        """
    ))

    cells.append(md(
        """
        ## 5. 阶段二补充：单特征候选调参

        固定参数结果出来后，我没有对所有特征和模型做全量穷举，而是筛选出 14 个更值得尝试的候选。筛选依据主要有两类：一类是 macro-F1 靠前，另一类是少数类 recall 有明显贡献。高维 KNN 的距离搜索比较耗时，所以使用 train 内部 holdout/subsample；低维组合则尽量用 3-fold CV。

        这一阶段的结论是：HOG 仍然是最强单特征，Color 对少数类召回尤其有帮助，Texture 低维但稳定，GIST 和 BoF 单独不一定强，但它们可能在融合阶段提供补充信息。这些判断直接决定了下一阶段的特征组合顺序。
        """
    ))

    cells.append(code(
        """
        s2_tuned = read_csv(REPORT_DIR / "stage2_tuned_best_by_feature.csv")
        s2_tuned["best_params_short"] = s2_tuned["best_params"].map(lambda x: short_text(x, 95))

        display(Markdown("**单特征调参后的最佳结果**"))
        display(s2_tuned[[
            "feature_id", "model", "feature_dim",
            "fixed_validation_macro_f1", "validation_macro_f1",
            "minority_recall_mean", "search_strategy",
            "best_params_short"
        ]].sort_values("validation_macro_f1", ascending=False).reset_index(drop=True))

        plot_df = s2_tuned.sort_values("validation_macro_f1", ascending=False).copy()
        x = np.arange(len(plot_df))
        width = 0.36

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        axes[0].bar(x - width / 2, plot_df["fixed_validation_macro_f1"], width, label="fixed")
        axes[0].bar(x + width / 2, plot_df["validation_macro_f1"], width, label="tuned")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(plot_df["feature_id"], rotation=25)
        axes[0].set_title("Fixed vs Tuned Single-Feature Results")
        axes[0].set_ylabel("Validation Macro-F1")
        axes[0].legend()

        sns.scatterplot(
            data=plot_df,
            x="validation_macro_f1", y="minority_recall_mean",
            hue="feature_id", s=120, ax=axes[1]
        )
        for _, row in plot_df.iterrows():
            axes[1].text(row["validation_macro_f1"] + 0.002, row["minority_recall_mean"], row["feature_id"], fontsize=9)
        axes[1].set_title("Tuned Feature Trade-off")
        axes[1].set_xlabel("Validation Macro-F1")
        axes[1].set_ylabel("Minority Recall Mean")
        axes[1].legend_.remove()

        plt.tight_layout()
        plt.show()
        """
    ))

    cells.append(md(
        """
        ## 6. 阶段三：自适应特征融合与模型广筛

        单特征阶段给出的信息很明确：HOG 是融合起点，Color 和 Texture 必须加入，GIST、BoF、Shape 需要通过融合实验验证。于是我没有直接固定一个全量特征组合，而是从 HOG 出发，逐步加入不同特征，并保留若干参考组合。

        这一阶段共完成 14 个特征组合 × 7 类模型的广筛。模型包括 Logistic Regression、KNN、Linear SVM、Decision Tree、Random Forest、XGBoost 和 LightGBM。广筛结果显示，LightGBM 对这类高维混合手工特征最适应，C11 全量手工特征的 validation macro-F1 最高；同时，`F3+F1+F2` 已经能达到很强的水平，说明 HOG、颜色和纹理是主干信息。
        """
    ))

    cells.append(code(
        """
        s3_screen = read_csv(REPORT_DIR / "stage3_model_screening.csv")
        s3_best_combo = read_csv(REPORT_DIR / "stage3_best_by_combo.csv")
        s3_best_model = read_csv(REPORT_DIR / "stage3_best_by_model.csv")
        s3_add = read_csv(REPORT_DIR / "stage3_feature_addition_summary.csv")

        display(Markdown("**阶段三广筛 Top 10**"))
        display(s3_screen.sort_values("validation_macro_f1", ascending=False)[[
            "combo_id", "feature_ids", "model", "feature_dim",
            "validation_accuracy", "validation_macro_f1", "minority_recall_mean"
        ]].head(10).reset_index(drop=True))

        display(Markdown("**各模型族的最佳结果**"))
        display(s3_best_model[[
            "model", "combo_id", "feature_ids", "validation_macro_f1", "minority_recall_mean"
        ]].sort_values("validation_macro_f1", ascending=False).reset_index(drop=True))

        heat = s3_screen.pivot_table(
            index="combo_id", columns="model", values="validation_macro_f1", aggfunc="max"
        )
        row_order = s3_best_combo.sort_values("validation_macro_f1", ascending=False)["combo_id"]
        heat = heat.loc[row_order]

        fig, axes = plt.subplots(1, 2, figsize=(17, 6.2))
        sns.heatmap(heat, cmap="YlGnBu", annot=True, fmt=".3f", linewidths=0.4, ax=axes[0])
        axes[0].set_title("Stage 3 Model Screening Heatmap")
        axes[0].set_xlabel("Model")
        axes[0].set_ylabel("Feature Combination")

        add_plot = s3_add.sort_values("rank", ascending=False)
        axes[1].plot(add_plot["validation_macro_f1"], add_plot["feature_ids"], marker="o", label="macro-F1")
        axes[1].plot(add_plot["minority_recall_mean"], add_plot["feature_ids"], marker="s", label="minority recall")
        axes[1].set_title("Feature Fusion Path")
        axes[1].set_xlabel("Score")
        axes[1].set_ylabel("")
        axes[1].legend()
        axes[1].grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        plt.show()
        """
    ))

    cells.append(md(
        """
        ## 7. 阶段四之一：强模型精调

        阶段三确认 LightGBM 和 XGBoost 是最有竞争力的强模型后，我只对短名单里的强候选做有限精调。搜索时保留阶段三的基准参数，再随机采样少量候选；搜索只在 train 内部 holdout 上进行，validation 仍然作为统一比较标准。

        这一步给出的一个重要经验是：盲目扩大调参并不一定带来更高 validation macro-F1。第一轮精调没有稳定刷新阶段三最优结果，反而提示我后面应该更关注类别权重和特征消融，而不是继续堆搜索空间。
        """
    ))

    cells.append(code(
        """
        s4_search = read_csv(REPORT_DIR / "stage4_boosting_tuning_search.csv")
        s4_final = read_csv(REPORT_DIR / "stage4_boosting_tuning_final.csv")
        s4_final["best_params_short"] = s4_final["best_params"].map(lambda x: short_text(x, 90))

        display(Markdown("**强模型精调后的 validation 结果**"))
        display(s4_final[[
            "candidate_id", "model", "combo_label", "sample_weight_mode",
            "stage3_macro_f1", "validation_macro_f1", "macro_f1_delta_vs_stage3",
            "minority_recall_mean", "best_params_short"
        ]].sort_values("validation_macro_f1", ascending=False).reset_index(drop=True))

        fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
        sns.stripplot(
            data=s4_search,
            x="model", y="inner_macro_f1",
            hue="candidate_id", dodge=True, ax=axes[0], size=7
        )
        axes[0].set_title("Inner Holdout Search Trials")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Inner Macro-F1")
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

        compare = s4_final.sort_values("validation_macro_f1", ascending=False).head(6).copy()
        x = np.arange(len(compare))
        width = 0.36
        axes[1].bar(x - width / 2, compare["stage3_macro_f1"], width, label="stage3 baseline")
        axes[1].bar(x + width / 2, compare["validation_macro_f1"], width, label="tuned")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(compare["candidate_id"], rotation=35, ha="right")
        axes[1].set_title("Stage 3 Baseline vs Tuned Final")
        axes[1].set_ylabel("Validation Macro-F1")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
        """
    ))

    cells.append(md(
        """
        ## 8. 阶段四之二：LightGBM 消融与类别不平衡处理

        在强模型精调没有明显突破后，我固定阶段三表现稳定的 LightGBM 参数，转向两个更有解释价值的问题：第一，哪些特征真的给最终模型带来贡献；第二，类别不平衡应该怎样处理。

        类别权重比较了 `none`、`balanced` 和 `sqrt_balanced`。其中 `sqrt_balanced` 是更温和的加权方式，它不像直接 balanced 那样把少数类权重放得过大。结果显示，C11 full + `sqrt_balanced` 在 validation 上取得 0.8131 的 macro-F1，同时保持少数类 recall。特征消融方面，去掉 BoF 后 macro-F1 和少数类 recall 都下降，说明局部关键点词袋虽然单独不强，但在融合模型里确实有价值；Shape 的贡献小，但在全量组合里仍然是正贡献。
        """
    ))

    cells.append(code(
        """
        s4_ab = read_csv(REPORT_DIR / "stage4_lgbm_ablation_imbalance.csv")
        ablation = s4_ab[s4_ab["experiment_type"].eq("ablation")].copy()
        imbalance = s4_ab[s4_ab["experiment_type"].eq("imbalance")].copy()

        display(Markdown("**LightGBM 特征消融结果**"))
        display(ablation[[
            "combo_id", "feature_dim_raw", "sample_weight_mode",
            "validation_macro_f1", "minority_recall_mean",
            "delta_macro_f1_vs_C11_none", "recall_trash", "recall_battery", "recall_biological"
        ]].sort_values("validation_macro_f1", ascending=False).reset_index(drop=True))

        display(Markdown("**类别不平衡处理结果**"))
        display(imbalance[[
            "combo_id", "sample_weight_mode", "validation_macro_f1",
            "minority_recall_mean", "recall_trash", "recall_battery", "recall_biological"
        ]].sort_values(["combo_id", "validation_macro_f1"], ascending=[True, False]).reset_index(drop=True))

        fig, axes = plt.subplots(1, 2, figsize=(15, 4.9))
        abl_plot = ablation.sort_values("validation_macro_f1", ascending=True)
        axes[0].barh(abl_plot["combo_id"], abl_plot["validation_macro_f1"], color="#4C78A8", label="macro-F1")
        axes[0].scatter(abl_plot["minority_recall_mean"], abl_plot["combo_id"], color="#F58518", label="minority recall", zorder=3)
        axes[0].set_title("LightGBM Ablation")
        axes[0].set_xlabel("Score")
        axes[0].legend()

        imb_plot = imbalance.sort_values(["combo_id", "sample_weight_mode"])
        sns.barplot(
            data=imb_plot,
            x="combo_id", y="validation_macro_f1",
            hue="sample_weight_mode", ax=axes[1]
        )
        axes[1].set_title("Weight Mode Comparison")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Validation Macro-F1")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].legend(title="weight")

        plt.tight_layout()
        plt.show()
        """
    ))

    cells.append(md(
        """
        ## 9. 阶段四之三：Stacking 集成

        当单个 LightGBM 已经接近瓶颈后，我尝试用 stacking 学习不同模型之间的互补关系。这里的 base learner 不是随便堆模型，而是刻意保留不同偏好的模型：线性模型看全局线性边界，KNN 看局部邻域结构，Random Forest 看低维颜色/纹理/结构组合，LightGBM 则作为强主模型和 anchor。

        meta learner 使用 Logistic Regression。训练时先在 train split 上做 5-fold OOF 预测，避免 meta learner 直接看到 base learner 在训练样本上的过拟合输出。validation 只用于最终比较不同 ensemble 方案。结果表明，small-only stacking 不能超过强 LightGBM，但加入 LGBM anchor 后，stacking 的 validation macro-F1 提升到 0.8178，成为最终封存方案，也就是本文所称的 Garbage Net。
        """
    ))

    cells.append(code(
        """
        s4_stack_summary = read_json(REPORT_DIR / "stage4_stacking_summary.json")
        s4_base = read_csv(REPORT_DIR / "stage4_stacking_base_summary.csv")
        s4_stack = read_csv(REPORT_DIR / "stage4_stacking_results.csv")

        display(Markdown("**Stacking base learners**"))
        display(s4_base[[
            "base_id", "group", "model_type", "combo_id", "feature_dim_raw",
            "output_type", "validation_macro_f1", "validation_minority_recall_mean"
        ]].sort_values("validation_macro_f1", ascending=False).reset_index(drop=True))

        display(Markdown("**Ensemble 方案对比**"))
        stack_display = s4_stack[[
            "ensemble_id", "ensemble_type", "num_base_models", "num_meta_features",
            "meta_model", "meta_C", "meta_weight_mode",
            "validation_accuracy", "validation_macro_f1",
            "validation_minority_recall_mean", "delta_macro_f1_vs_stage4_best"
        ]].sort_values("validation_macro_f1", ascending=False).reset_index(drop=True)
        display(stack_display)

        best_stage4 = s4_stack_summary["stage4_best_reference"]["validation_macro_f1"]

        fig, axes = plt.subplots(1, 2, figsize=(15, 4.9))
        sns.barplot(
            data=s4_base.sort_values("validation_macro_f1", ascending=False),
            x="validation_macro_f1", y="base_id", ax=axes[0], color="#4C78A8"
        )
        axes[0].set_title("Base Learner Validation Macro-F1")
        axes[0].set_xlabel("Validation Macro-F1")
        axes[0].set_ylabel("")

        sns.barplot(
            data=s4_stack.sort_values("validation_macro_f1", ascending=False),
            x="validation_macro_f1", y="ensemble_id", ax=axes[1], color="#54A24B"
        )
        axes[1].axvline(best_stage4, color="#E45756", linestyle="--", label="best single LGBM")
        axes[1].set_title("Ensemble Validation Macro-F1")
        axes[1].set_xlabel("Validation Macro-F1")
        axes[1].set_ylabel("")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        best_ensemble = s4_stack_summary["best_ensemble"]
        display(pd.DataFrame([{
            "Garbage Net validation macro-F1": best_ensemble["validation_macro_f1"],
            "Garbage Net validation minority recall": best_ensemble["validation_minority_recall_mean"],
            "base models": best_ensemble["num_base_models"],
            "meta features": best_ensemble["num_meta_features"],
            "meta learner": best_ensemble["meta_model"],
        }]))
        """
    ))

    cells.append(md(
        """
        ## 10. 最终 test 集评价

        最终模型在 validation 阶段已经确定为 `final_stack_small_plus_lgbm_anchor`，也就是 Garbage Net。最终评价时，我把 train 和 validation 合并作为最终训练数据，重新生成 5-fold OOF 预测训练 meta learner，再让每个 base learner 用完整 train+validation 重训，最后对 test 集预测一次。

        从 test 结果看，Garbage Net 的 macro-F1 为 0.8097，仍然优于单个 LightGBM anchor。validation 到 test 有小幅回落，但幅度不大，说明模型选择没有严重依赖 validation 的偶然性。逐类结果中，battery、biological、clothes 等类别表现较好；trash 的 recall 只有 0.6029，是最终模型最明显的短板。
        """
    ))

    cells.append(code(
        """
        final_summary = read_json(REPORT_DIR / "final_test_evaluation_summary.json")
        final_models = read_csv(REPORT_DIR / "final_model_results.csv")
        final_base = read_csv(REPORT_DIR / "final_base_results.csv")
        final_per_class = read_csv(REPORT_DIR / "final_per_class_metrics.csv")
        final_errors = read_csv(REPORT_DIR / "final_error_pairs.csv")

        best = final_summary["best_final_model"]
        key_metrics = pd.DataFrame([{
            "model_name": "Garbage Net",
            "model_id": best["model_id"],
            "test_accuracy": best["test_accuracy"],
            "test_macro_f1": best["test_macro_f1"],
            "test_weighted_f1": best["test_weighted_f1"],
            "test_minority_recall_mean": best["test_minority_recall_mean"],
            "test_recall_trash": best["test_recall_trash"],
            "test_recall_battery": best["test_recall_battery"],
            "test_recall_biological": best["test_recall_biological"],
        }])
        display(Markdown("**Garbage Net 最终 test 指标**"))
        display(key_metrics)

        display(Markdown("**最终 ensemble 方案 test 对比**"))
        display(final_models[[
            "model_id", "model_type", "num_base_models",
            "test_accuracy", "test_macro_f1", "test_weighted_f1",
            "test_minority_recall_mean"
        ]].sort_values("test_macro_f1", ascending=False).reset_index(drop=True))

        stack_class = final_per_class[
            (final_per_class["model_id"].eq("final_stack_small_plus_lgbm_anchor")) &
            (final_per_class["split"].eq("test"))
        ].copy()
        display(Markdown("**Garbage Net 逐类别 test 指标**"))
        display(stack_class[["class_name", "support", "precision", "recall", "f1"]].reset_index(drop=True))

        display(Markdown("**主要错误对 Top 12**"))
        display(final_errors.sort_values("count", ascending=False).head(12).reset_index(drop=True))

        display(Markdown("**最终评价图表**"))
        for svg_name in [
            "final_model_comparison.svg",
            "final_stack_confusion_matrix.svg",
            "final_stack_per_class_performance.svg",
        ]:
            svg_path = FIGURE_DIR / svg_name
            if svg_path.exists():
                display(SVG(filename=str(svg_path)))
        """
    ))

    cells.append(md(
        """
        ## 11. 错误分析

        最终错误主要集中在几组视觉边界比较模糊的类别上。plastic 和 glass 容易互相混淆，因为它们都可能透明、反光，并且在背景复杂时局部颜色和边缘特征相近。cardboard、paper、plastic 之间也有一部分误分，尤其是包装纸盒、纸袋或塑料包装的形态接近时。trash 的问题更明显：它样本最少，类内变化又大，所以 recall 明显低于其他类别。

        这些错误说明，继续微调分类器参数的收益可能有限。更直接的改进方向是补充少数类样本、增加针对 trash 的数据增强，或者在传统特征前加入更稳定的主体区域提取，减少背景对颜色和纹理统计的干扰。
        """
    ))

    cells.append(code(
        """
        error_pairs = read_csv(REPORT_DIR / "final_error_pairs.csv").sort_values("count", ascending=False)
        error_examples = read_csv(REPORT_DIR / "final_error_examples.csv")

        fig, ax = plt.subplots(figsize=(9, 5.2))
        top_pairs = error_pairs.head(12).copy()
        top_pairs["pair"] = top_pairs["true_label"] + " -> " + top_pairs["predicted_label"]
        sns.barplot(data=top_pairs, x="count", y="pair", ax=ax, color="#E45756")
        ax.set_title("Top Error Pairs")
        ax.set_xlabel("Errors")
        ax.set_ylabel("")
        bar_labels(ax, fmt="{:.0f}")
        plt.tight_layout()
        plt.show()

        display(Markdown("**高置信度误分类样本示例**"))
        examples = error_examples.head(9).copy()
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        for ax, (_, row) in zip(axes.ravel(), examples.iterrows()):
            img_path = Path(row["path"])
            if img_path.exists():
                ax.imshow(PILImage.open(img_path))
            ax.set_title(
                f"true: {row['true_label']}\\npred: {row['predicted_label']} ({row['confidence']:.2f})",
                fontsize=10
            )
            ax.axis("off")
        for ax in axes.ravel()[len(examples):]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        """
    ))

    cells.append(md(
        """
        ## 12. 与已有传统机器学习和神经网络结果的对比

        为了给最终结果一个参照，我另外整理了几类传统机器学习文献方案的本地复刻结果，包括 SIFT BoF + RBF SVM、手工特征 + LightGBM/XGBoost、RGB 像素 + PCA + Random Forest/SVM、HOG + 纹理 + 线性 SVM 等。它们只作为横向参考，不参与当前主模型的选择，也不使用 test 集。

        和这些传统候选相比，本项目的提升主要来自三个方面：第一，不依赖单一特征，而是经过单特征基线和融合广筛后确定特征组合；第二，对类别不平衡采用了更温和的 `sqrt_balanced`；第三，用 stacking 把多个小模型和强 LightGBM anchor 的互补信息合在一起。

        和神经网络方法相比，传统机器学习路线的优势是结构清楚、训练成本较低、特征贡献更容易解释；劣势是表达能力依赖手工特征设计，面对背景复杂、主体形态变化大、类间视觉差异细微的样本时，通常不如预训练深度模型灵活。因此我不会把 Garbage Net 说成全面替代深度学习，而是把它作为一条可解释、可复现实验链路下得到的强传统机器学习基线。
        """
    ))

    cells.append(code(
        """
        lit = read_csv(REPORT_DIR / "literature_traditional_results.csv")
        lit_ok = lit[lit["status"].eq("ok")].copy()
        lit_display = lit_ok[[
            "candidate_id", "source_short", "model_label", "feature_dim",
            "validation_accuracy", "validation_macro_f1",
            "validation_minority_recall_mean", "actual_train_seconds"
        ]].sort_values("validation_macro_f1", ascending=False)

        display(Markdown("**传统文献候选的本地 validation 复刻结果**"))
        display(lit_display.reset_index(drop=True))

        comparison = pd.concat([
            lit_display[["candidate_id", "validation_macro_f1", "validation_minority_recall_mean"]]
            .rename(columns={"candidate_id": "method"}),
            pd.DataFrame([{
                "method": "Garbage Net (validation)",
                "validation_macro_f1": s4_stack_summary["best_ensemble"]["validation_macro_f1"],
                "validation_minority_recall_mean": s4_stack_summary["best_ensemble"]["validation_minority_recall_mean"],
            }])
        ], ignore_index=True).sort_values("validation_macro_f1", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5.2))
        sns.barplot(data=comparison, x="validation_macro_f1", y="method", ax=ax, color="#4C78A8")
        ax.set_title("Traditional References vs Garbage Net on Validation")
        ax.set_xlabel("Validation Macro-F1")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.show()
        """
    ))

    cells.append(md(
        """
        ## 13. 总结与反思

        这次实验里最有价值的不是单个最终分数，而是逐步排除和确认的过程。单特征阶段说明 HOG 是最强起点，但 Color 和 Texture 对少数类很关键；特征融合阶段说明 C11 全量手工特征最稳，BoF 单独表现一般但在融合中有贡献；强模型阶段说明 LightGBM 是最合适的主模型；类别权重实验说明 `sqrt_balanced` 比直接加重少数类更稳；最后 stacking 证明，小模型本身不一定强，但它们的预测可以为强模型 anchor 提供补充信息。

        最终的 Garbage Net 在 test 集上取得 macro-F1 = 0.8097，相比单个 LightGBM anchor 仍有提升。不过模型也有明显短板：trash 类 recall 偏低，plastic/glass 等材质接近类别仍然容易混淆。如果继续改进，我会优先从少数类数据补充、主体区域提取和针对易混淆类别的特征增强入手，而不是继续盲目扩大参数搜索。
        """
    ))

    cells.append(md(
        """
        ## 附录 A：可复现性说明

        本 notebook 使用项目内已经固定下来的数据划分、特征缓存和实验结果文件生成。所有代码单元都使用相对路径读取数据，不依赖本地绝对路径。随机划分、特征提取、模型筛选、调参、stacking 和最终 test 评价的中间结果都保存在 `data/processed/` 下。
        """
    ))

    cells.append(code(
        """
        artifacts = [
            "data/processed/splits/split_summary.csv",
            "data/processed/feature_reports/feature_extraction_report.csv",
            "data/processed/model_reports/stage2_single_feature_baselines.csv",
            "data/processed/model_reports/stage2_tuned_best_by_feature.csv",
            "data/processed/model_reports/stage3_model_screening.csv",
            "data/processed/model_reports/stage4_lgbm_ablation_imbalance.csv",
            "data/processed/model_reports/stage4_stacking_results.csv",
            "data/processed/model_reports/final_test_evaluation_summary.json",
            "data/processed/model_reports/final_per_class_metrics.csv",
        ]
        artifact_table = pd.DataFrame({
            "artifact": artifacts,
            "exists": [Path(p).exists() for p in artifacts],
        })
        display(artifact_table)

        deps_path = REPORT_DIR / "final_dependency_versions.json"
        if deps_path.exists():
            deps = read_json(deps_path)
            display(Markdown("**最终评价时记录的依赖版本**"))
            display(pd.DataFrame([deps]))
        """
    ))

    cells.insert(2, md(
        """
        ## 2. 论文综述与方法定位

        图像垃圾分类研究大致经历了三类技术路线：基于手工特征的传统机器学习、端到端深度神经网络，以及深度特征提取器与传统分类器结合的混合方法。本项目位于第一类路线，即先将图像转换为可解释的数值特征，再使用传统机器学习模型完成分类。该路线的优势在于实验链路清晰、模型训练成本可控、特征贡献便于解释，适合展示完整的建模、试错、调参与决策过程。
        """
    ))

    expanded_markdown_sources = [
        """
        # Garbage Net:一种基于手工特征与传统机器学习方法的垃圾图像分类模型

        ## 摘要

        本项目研究 10 类垃圾图像分类任务，类别包括 battery、biological、cardboard、clothes、glass、metal、paper、plastic、shoes 和 trash。任务难点主要体现在三个方面：第一，图像并非简单的白底单物体场景，背景、光照、主体尺寸和拍摄角度存在较大变化；第二，类别分布不均衡，trash 仅有 453 张，而 clothes 达到 1892 张，若只追求 accuracy，模型容易偏向样本数较多的类别；第三，部分类别存在真实视觉边界模糊，例如 plastic 与 glass 都可能透明或反光，paper、cardboard 与 plastic 包装也可能在颜色和形状上接近。

        本项目没有采用端到端神经网络训练，而是构建了一条完整的传统机器学习建模路线：首先对数据集进行质量审查与固定分层划分，然后从图像中提取 F0-F6 七组特征，包括原始像素基线、颜色特征、纹理特征、HOG 梯度方向特征、形状特征、GIST 全局结构特征以及 SIFT/ORB Bag-of-Features 局部词袋特征；随后通过固定参数单特征基线、单特征候选调参、特征融合广筛、强模型精调、LightGBM 消融、类别不平衡处理和 stacking 集成逐步筛选最终方案。整个实验过程使用 validation 集进行模型选择和参数决策，test 集在最终模型封存后只使用一次。

        本研究将最终选出的最优 stacking 集成模型命名为 **Garbage Net**。Garbage Net 由 6 个 base learner 组成，分别覆盖 Logistic Regression、Linear SVM、Random Forest、KNN 和两个 LightGBM 变体，再使用 Logistic Regression 作为 meta learner 学习各 base learner 输出之间的互补关系。在最终 test 集上，Garbage Net 取得 accuracy = 0.8238、macro-F1 = 0.8097、weighted-F1 = 0.8228，少数类 recall 均值为 0.7420。相较单个 LightGBM C11 anchor 的 test macro-F1 = 0.7950，Garbage Net 仍保持可观察的提升，说明 stacking 对传统手工特征模型具有实际增益。
        """,
        """
        ## 1. 实验设计与整体流程

        本项目的核心目标不是直接给出一个最终分类器，而是构建一条可复现、可解释、能展示试错过程的建模链路。图像首先被转换为固定长度的特征向量，然后输入不同类型的传统机器学习模型。每个阶段的设计都对应一个具体问题：单一特征是否具有独立分类能力，哪些特征在少数类上更有价值，特征融合是否真正带来增益，强模型是否优于简单模型，类别权重是否改善不平衡问题，以及 stacking 是否能够利用不同模型之间的互补信息。

        完整实验流程包括 9 个步骤。第一步是数据审查与固定划分，检查图像能否正常读取、类别数量是否一致、train/validation/test 是否保持分层比例。第二步是图像预处理，统一使用 `standardized_256` 图像版本，并根据特征需要转换为 RGB、BGR、HSV、Gray 和二值 mask 表示。第三步是特征提取，把每张图像表示为 F0-F6 不同特征组，其中 F1-F6 构成主要手工特征空间。第四步是固定参数单特征基线，用 Logistic Regression、KNN、Linear SVM 和 Decision Tree 观察每组特征的基础可分性。第五步是单特征候选调参，只对阶段四前有价值的组合做轻量搜索，避免无意义穷举。

        第六步是自适应特征融合与模型广筛。实验从 HOG 出发，逐步加入 Color、Texture、GIST、BoF 和 Shape，并在 14 个特征组合与 7 类模型上完成 98 组 validation 评价。第七步是强模型精调，主要比较 LightGBM 和 XGBoost 在短名单组合上的调参收益。第八步是 LightGBM 消融与类别不平衡处理，重点比较 no BoF、no Shape、no GIST、低维组合以及 `none`、`balanced`、`sqrt_balanced` 三类 sample weight。第九步是 stacking 集成，在单模型接近瓶颈后，用 5-fold OOF 预测训练 meta learner，最终确定 Garbage Net。

        指标设计以 macro-F1 为主，因为 macro-F1 对每个类别等权，不会被 clothes、glass、plastic 等大类主导。accuracy 和 weighted-F1 用于观察整体分类正确率与样本加权表现；少数类 recall 均值用于衡量 trash、battery、biological 三类的召回能力。validation 集承担模型选择、调参和消融比较职责；test 集只在最终模型确定后使用，避免出现面向 test 的反复调参。
        """,
        """
        ## 2. 论文综述与方法定位

        图像垃圾分类是计算机视觉在智能回收和自动化垃圾分拣中的典型应用。已有研究主要采用三类方法：第一类是手工特征与传统机器学习分类器，第二类是端到端卷积神经网络或迁移学习模型，第三类是使用深度模型提取 embedding 后再接传统分类器的混合方法。本项目聚焦第一类路线，原因在于课程大作业的重点是展示完整建模过程，包括特征构造、特征选择、模型筛选、参数调整和错误分析，而不是只呈现一个黑箱深度模型的最终准确率。

        Yang 和 Thung（2016）的 [Classification of Trash for Recyclability Status](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf) 是较早的垃圾图像分类研究之一。该研究基于 TrashNet，将垃圾划分为 glass、paper、metal、plastic、cardboard 和 trash 六类，使用 SIFT 局部关键点、Bag of Features 视觉词袋和 RBF 核 SVM 进行分类，并与 CNN 进行比较。该工作对本项目有两点启发：其一，垃圾图像中材料纹理、局部关键点和形状边缘具有重要判别信息，因此本项目保留了 SIFT/ORB BoF 与 HOG；其二，trash 类样本少且视觉差异大，容易成为召回短板，因此本项目在评价指标中单独关注少数类 recall，而不只看 accuracy。

        Nguyen 等人的 [Towards Accurate and Efficient Waste Image Classification](https://arxiv.org/abs/2510.21833) 系统比较了手工特征、深度学习和混合方案，手工特征部分覆盖颜色、形状、纹理、关键点和 GIST 等描述子，并使用传统分类器进行建模。这一思路直接指导了本项目的特征体系设计：F1 使用 HSV/BGR 颜色直方图和统计量，F2 使用 GLCM 与 LBP 纹理，F4 使用几何形状和 Hu moments，F5 使用 GIST，全局和局部信息同时进入候选空间。Sen 等人的 [Feature Engineering is Not Dead](https://arxiv.org/abs/2507.13772) 进一步说明，在需要可解释性和较低计算成本的图像分类任务中，HOG、LBP 等手工特征与 SVM 等传统模型仍具有研究价值。Mboli 和 Ogungbemi（2026）的 [AI-Enabled Waste Classification](https://arxiv.org/abs/2601.22418) 则比较了传统机器学习、PCA 与深度学习模型，提示降维并不一定稳定提升传统模型，特征选择必须结合具体模型和数据验证。

        本项目不把非线性核 SVM 作为主体模型继续深入，原因包括研究重复性和计算成本两方面。首先，Yang 和 Thung 已经在垃圾分类任务中系统尝试了 SIFT/BoF + RBF-SVM，并给出了 RBF 核、C=1000、gamma=0.5 的设置；本项目的文献复刻也实现了 `yang_thung_2016_sift_bof_rbf_svm`，validation macro-F1 为 0.4240，少数类 recall 均值为 0.4599，明显低于后续 LightGBM、XGBoost 和 stacking 方案。其次，核 SVM 的训练和预测通常依赖样本间核矩阵及支持向量计算，在本项目 8581 个 train 样本、3165 维全量手工特征、98 组广筛实验的规模下，不适合作为大规模主线搜索模型。相比之下，线性 SVM 可通过 SGDClassifier 扩展到高维特征，并能作为 stacking 中的异质 base learner。因此，本项目将 SVM 路线限制在线性核和文献复刻对照中，把主要搜索资源投入到更可扩展、表现更强的树集成模型与 stacking 集成上。
        """,
        """
        ## 3. 数据集介绍与固定划分

        数据集目录包含三个源文件夹：`original`、`standardized_256` 和 `standardized_384`。`original` 保存原始图像版本，主要用于数据来源追溯、质量检查和示例展示；该版本保留了原图尺寸差异，适合观察真实图像状态，但不适合作为传统机器学习的直接输入，因为尺寸不统一会导致特征长度不一致。`standardized_384` 是较高分辨率的标准化版本，适合需要更多空间细节的模型，例如后续如果扩展到深度学习或更高分辨率特征提取，可以作为候选输入。`standardized_256` 是本项目的正式建模版本，所有图像被统一到 256×256，既保留了足够的颜色、边缘、纹理和局部关键点信息，又能控制 HOG、GIST、BoF 等特征提取的计算成本。

        三个源文件夹都覆盖相同的 10 个类别，并对应 12259 张有效图像。正式建模选择 `standardized_256` 的原因有三点。第一，传统机器学习需要每张图像转换为固定长度向量，统一尺寸可以避免额外插值和裁剪策略在不同实验间不一致。第二，256×256 对 HOG 和 SIFT/ORB BoF 较为合适：HOG 可以在下采样到 128×128 后提取 1764 维梯度方向特征，SIFT/ORB 可以在可接受时间内形成视觉词袋。第三，384×384 虽然保留更多细节，但会显著增加关键点检测、局部描述子聚类和像素类 baseline 的计算量；在本项目强调分阶段多模型比较的设置下，256 版本更适合作为统一输入。

        类别分布存在明显不均衡。clothes 有 1892 张，glass 有 1736 张，plastic 有 1597 张，shoes 有 1449 张，cardboard 有 1411 张，paper 有 1336 张；相对少数类包括 metal 930 张、battery 756 张、biological 699 张，trash 只有 453 张。固定划分采用 train / validation / test = 70% / 15% / 15% 的分层策略，最终 train 为 8581 张、validation 为 1839 张、test 为 1839 张。每个类别在三个 split 中保持近似相同比例，例如 trash 被划分为 train 317、validation 68、test 68，battery 被划分为 train 530、validation 113、test 113。这样的固定划分保证所有模型和特征组合在完全相同的数据条件下比较。

        validation 集用于模型筛选、特征组合比较、参数调整和 stacking 方案选择；test 集不参与任何中间决策，只在最终模型 Garbage Net 确定后使用一次。由于类别不平衡较明显，模型评价不能仅依赖 accuracy。后续实验将 macro-F1 作为主指标，并在所有关键表格中同时输出少数类 recall 均值，用来观察模型是否只提升大类而牺牲少数类。
        """,
        """
        ## 4. 图像预处理与特征体系

        图像预处理的目标是把非结构化图像转换为稳定、可复现的结构化特征。正式输入来自 `standardized_256`，每张图像统一为 256×256。根据不同特征的需求，预处理阶段生成 RGB、BGR、HSV、Gray 和近似前景 mask 等表示。颜色特征主要使用 HSV 与 BGR；纹理、HOG 和 GIST 主要依赖灰度图；形状特征使用阈值化和边缘信息近似主体区域；SIFT/ORB BoF 则从局部关键点描述子出发，经训练集视觉词典转换为固定长度词频向量。

        特征体系分为七组。F0 是原始像素 baseline，包括 32×32 灰度像素 1024 维和 32×32 RGB 像素 3072 维，仅用于观察直接像素输入的最低基线。F1 是颜色特征，共 1039 维，由 HSV 统计量和 HSV/BGR 三维颜色直方图组成，主要描述材料颜色、亮度和饱和度分布。F2 是纹理特征，共 30 维，包括 GLCM 的 contrast、dissimilarity、homogeneity、energy、correlation 等统计量以及 LBP 直方图，用于描述纸张、纸板、布料、金属和塑料表面的局部模式。F3 是 HOG 特征，共 1764 维，用于捕捉边缘方向和局部轮廓结构，是单特征阶段表现最强的主干特征。

        F4 是形状特征，共 12 维，包括面积、周长、长宽比以及 Hu moments 等几何信息。虽然 F4 单独分类能力不强，但其维度低、成本小，可以作为全量融合中的补充项。F5 是 GIST 全局结构特征，共 64 维，描述图像整体空间布局和方向响应。F6 是 SIFT+ORB Bag-of-Features，共 256 维，其中 SIFT 视觉词典 128 维、ORB 视觉词典 128 维，主要描述局部关键点和材料局部结构。最终全量手工特征 C11 为 F1+F2+F4+F5+F3+F6，总维度 3165。

        所有与数据分布相关的处理都遵循防止数据泄漏原则。例如，StandardScaler、RobustScaler、Normalizer、PCA、k-means 视觉词典、模型参数搜索和 stacking OOF 生成都只在 train 或 train 内部划分上拟合，再应用到 validation/test。下面的特征检查表显示，各特征缓存均存在，train/validation/test 形状一致，nan 和 inf 计数均为 0，说明后续建模可以直接基于这些缓存展开。
        """,
        """
        ## 5. 阶段二：固定参数单特征基线

        固定参数单特征基线用于回答第一个建模问题：在不调参、不融合的条件下，每组特征本身是否包含有效分类信息。该阶段使用 7 个单特征组和 4 类传统模型，共形成 28 组实验。模型选择覆盖线性、距离、最大间隔和树模型四种基础假设：Logistic Regression 用于检验线性可分性；KNN 用于检验局部邻域结构；Linear SVM 通过 hinge loss 检验高维线性间隔；Decision Tree 用于检验单棵树对非线性规则的表达能力。

        固定参数设置保持简单且一致。Logistic Regression 使用 StandardScaler + `C=1`、`lbfgs`、`max_iter=1000`；KNN 使用 StandardScaler + `n_neighbors=5`、`weights=distance`、`metric=euclidean`；Linear SVM 使用 StandardScaler + `SGDClassifier(loss=hinge, alpha=0.0001, max_iter=1000)`；Decision Tree 使用默认树深度并固定 `random_state=42`。该阶段不追求最优参数，而是用统一规则判断特征价值。

        输出结果显示，固定参数阶段的最佳组合是 F3 HOG + KNN，validation macro-F1 = 0.5163，少数类 recall 均值 = 0.4344。第二名是 F5 GIST + KNN，macro-F1 = 0.5037，少数类 recall = 0.4984。F2 Texture + Logistic Regression 的 macro-F1 = 0.4956，少数类 recall = 0.5555，说明低维纹理特征虽然整体区分能力不是最高，但对少数类较敏感。F1 Color + KNN 的 macro-F1 = 0.4816，少数类 recall = 0.5470，同样显示颜色分布对 battery、biological、trash 等类别有补充作用。

        该阶段的结论是：HOG 是最强单特征，应作为后续融合起点；Color 和 Texture 在少数类召回上有明显价值，不能因为 macro-F1 略低而删除；GIST 维度低且表现稳定，可以作为全局结构补充；BoF 和 Shape 单独表现不突出，但需要在融合阶段继续验证是否具有互补性。散点图中的 macro-F1 与 minority recall 分布也显示，单一指标不足以支撑特征选择，必须同时观察整体表现和少数类表现。
        """,
        """
        ## 6. 阶段二补充：单特征候选调参

        固定参数结果提供了第一轮筛选依据，但固定参数可能低估某些特征的潜力。因此阶段二补充实验只对 14 个有价值候选进行轻量调参，而不是对所有特征、模型和参数做全量穷举。筛选标准包括：固定参数 macro-F1 排名前列、少数类 recall 表现较好、或在特征含义上可能与其他特征互补。这样既能展示调参过程，又能避免把大量计算浪费在明显较弱的组合上。

        调参策略根据特征维度和模型成本区别设计。高维 KNN 的距离计算较重，因此 HOG、Color、BoF、Gray pixel 等高维组合使用 train 内部 holdout 或 subsample；Texture、GIST 等低维组合使用 3-fold CV。SVM 搜索空间在试验中被收窄为线性 hinge + l2 方向，因为高维图像特征上的大规模 SVM 搜索非常耗时，且非线性核 SVM 已在文献中被充分尝试，不适合作为本项目后续主线。

        调参后，F3 HOG + KNN 从固定参数 macro-F1 = 0.5163 提升到 0.5612，最佳参数为 Normalizer + `k=9`、manhattan 距离、distance 权重，是最强单特征结果。F2 Texture + KNN 得到 macro-F1 = 0.5024，少数类 recall = 0.5081，说明纹理特征在低维空间中仍较稳定。F1 Color + KNN 的 macro-F1 = 0.4980，但少数类 recall = 0.5620，是单特征中少数类召回最突出的组合。F5 GIST + KNN 得到 macro-F1 = 0.4942；F6 BoF + KNN 虽然 macro-F1 只有 0.4324，但少数类 recall = 0.4957，说明局部词袋可能对部分少数类有补充价值。

        由此形成后续融合策略：以 HOG 为主干，优先加入 Color 和 Texture；GIST 作为低维全局结构补充；BoF 不因单独较弱而直接舍弃，而是在融合模型中验证其边际贡献；Shape 维度很低、成本小，也保留到消融实验中判断其最终价值。
        """,
        """
        ## 7. 阶段三：自适应特征融合与模型广筛

        阶段三的目标是从单特征判断进入多特征建模。实验没有直接把所有特征拼接后训练一个模型，而是采用自适应融合策略：先以阶段二表现最强的 HOG 为起点，再根据少数类贡献和特征互补性逐步加入 Color、Texture、GIST、BoF 和 Shape。这样可以观察每一次加入特征后的边际变化，并避免“全量特征最好”这一结论缺乏过程依据。

        本阶段共完成 14 个特征组合 × 7 类模型 = 98 组实验。模型族覆盖 Logistic Regression、KNN、Linear SVM、Decision Tree、Random Forest、XGBoost 和 LightGBM。线性模型用于提供可解释的高维基线；KNN 用于保留距离式局部结构判断；Decision Tree 和 Random Forest 用于观察树模型对非线性特征交互的处理能力；XGBoost 与 LightGBM 代表梯度提升树，是本阶段最重要的强模型候选。

        输出结果显示，LightGBM 明显领先。C11 全量手工特征 + LightGBM 的 validation macro-F1 = 0.8096，少数类 recall 均值 = 0.8158，是阶段三最优结果。去掉 Shape 的 F3+F1+F2+F5+F6 + LightGBM 得到 macro-F1 = 0.8071，少数类 recall = 0.8109；C10 no BoF + LightGBM 得到 macro-F1 = 0.8020，少数类 recall = 0.7961；仅使用 HOG+Color+Texture 的 LightGBM 已达到 macro-F1 = 0.8012，少数类 recall = 0.8006。这说明 HOG、颜色和纹理构成主要信息来源，而 BoF 和 Shape 则提供进一步增益。

        按模型族比较，LightGBM 的最佳 macro-F1 = 0.8096，XGBoost 次之，最佳 macro-F1 = 0.7980，Random Forest 最佳为 0.7049，Linear SVM 最佳为 0.6885，Logistic Regression 最佳为 0.6688，KNN 最佳为 0.5787，Decision Tree 最佳为 0.4705。该结果表明，高维混合手工特征包含复杂的非线性关系，单一线性模型和单棵树不足以充分利用这些信息；梯度提升树更适合处理不同尺度、不同语义来源的拼接特征。因此，后续阶段将 LightGBM 作为主模型，同时保留 XGBoost 和若干小模型作为对照或 stacking 候选。
        """,
        """
        ## 8. 阶段四之一：强模型精调

        阶段三已经确认 LightGBM 和 XGBoost 是最强模型族，因此阶段四首先对短名单中的强模型进行有限精调。调参没有采用无边界的大规模搜索，而是保留阶段三表现较好的参数作为基准，再随机采样少量候选。搜索过程只在 train 内部 holdout 上进行，validation 集仍用于统一最终比较。这一设计可以减少 validation 被反复使用导致的过拟合风险。

        LightGBM 搜索围绕 `n_estimators`、`learning_rate`、`num_leaves`、`subsample`、`colsample_bytree`、`reg_lambda` 和 sample weight 展开；XGBoost 搜索围绕 `max_depth`、`learning_rate`、`n_estimators`、`subsample`、`colsample_bytree` 和正则项展开。由于 XGBoost 单次训练成本较高，trial 数量少于 LightGBM。所有候选最终都回到完整 train split 上重训，再在 validation split 上评价。

        精调结果显示，第一轮随机调参并没有刷新阶段三最优 macro-F1。LGBM_A_C11_full 的 validation macro-F1 = 0.8036，少数类 recall = 0.8272，相比阶段三 C11 LightGBM 的 0.8096 下降 0.0060，但少数类 recall 有所提高。LGBM_B_no_shape 的 macro-F1 = 0.8000，少数类 recall = 0.8302，是本轮少数类 recall 最高的方案。XGB_A_C11_full 的 macro-F1 = 0.7996，较其阶段三基线小幅提升 0.0022，但少数类 recall = 0.7795，低于 LightGBM。XGB_B_no_shape 的 macro-F1 = 0.7959，也未超过 LightGBM。

        该阶段的重要结论是，更多调参不必然带来更好的 validation macro-F1。强模型已经接近阶段三设定下的性能上限后，盲目扩大随机搜索空间可能只是在 macro-F1 和少数类 recall 之间移动权衡。后续实验因此转向更具解释性的两个方向：固定 LightGBM 主参数后做特征消融，以及系统比较类别不平衡处理方式。
        """,
        """
        ## 9. 阶段四之二：LightGBM 消融与类别不平衡处理

        本阶段固定阶段三表现稳定的 LightGBM 参数，即 `n_estimators=460`、`num_leaves=31`、`learning_rate=0.05`、`subsample=0.85`、`colsample_bytree=0.75`、`reg_lambda=1.0`，再比较不同特征组合和类别权重策略。这样可以把变量集中在“特征是否有贡献”和“权重是否改善不平衡”两个问题上，而不是让模型参数变化干扰解释。

        特征消融结果显示，C11 full + balanced 的 validation macro-F1 = 0.8096，少数类 recall = 0.8158，作为消融基准。去掉 Shape 后 macro-F1 = 0.8071，下降 0.0025，说明 Shape 贡献较小但为正。去掉 BoF 后 macro-F1 = 0.8020，少数类 recall = 0.7961，较 C11 明显下降，说明 SIFT/ORB 局部词袋虽然单独表现一般，但在全量模型中提供了有用的局部结构补充。去掉 GIST 后 macro-F1 = 0.8054，但少数类 recall = 0.8221，说明 GIST 对整体 macro-F1 有帮助，但不同少数类之间存在取舍。仅使用 HOG+Color+Texture 的 macro-F1 = 0.8012，已经接近 C11，但仍低于全量特征。

        类别不平衡处理比较了 `none`、`balanced` 和 `sqrt_balanced`。C11 full + none 的 validation macro-F1 = 0.8086，少数类 recall = 0.7888；C11 full + balanced 的 macro-F1 = 0.8096，少数类 recall = 0.8158；C11 full + sqrt_balanced 的 macro-F1 = 0.8131，少数类 recall = 0.8158。`sqrt_balanced` 对 balanced 权重开方后再归一化，相当于保留少数类补偿但降低极端权重，因此在本数据集上取得了更好的总体平衡。

        该阶段最终确定 LightGBM C11 full + `sqrt_balanced` 作为最强单模型参考，其 validation macro-F1 = 0.8131。消融结果也为 stacking 设计提供依据：BoF、Shape、GIST 等特征可以保留在 anchor 或部分小模型中，但不必强制所有 base learner 都使用全量特征；低维 Color+Texture+GIST 虽然 macro-F1 较低，但少数类 recall 较高，适合作为异质 base learner。
        """,
        """
        ## 10. 阶段四之三：Stacking 集成

        当单个 LightGBM 的 validation macro-F1 达到 0.8131 后，继续调参的边际收益已经变小。因此，本阶段尝试 stacking 集成，通过学习不同模型输出之间的互补关系进一步提升性能。Stacking 的关键不是简单堆模型，而是构造具有差异性的 base learner，使它们在特征空间、模型假设和错误模式上尽量不同。

        本项目设计了 6 个 base learner。`lr_c11_balanced` 使用 C11 全量手工特征和 Logistic Regression，validation macro-F1 = 0.6639，提供线性概率输出。`sgd_svm_hct_balanced` 使用 HOG+Color+Texture 和线性 SVM，validation macro-F1 = 0.6139，提供最大间隔方向的 decision 输出。`rf_lowdim_balanced` 使用 Color+Texture+GIST 低维组合和 Random Forest，validation macro-F1 = 0.7035，少数类 recall = 0.7635，对少数类有补充价值。`knn_compact_gist_texture_shape` 使用 GIST+Texture+Shape 106 维紧凑特征，validation macro-F1 = 0.5492，提供局部邻域视角。`lgbm_small_hct_sqrt` 使用 HOG+Color+Texture，validation macro-F1 = 0.7899，是紧凑强模型。`lgbm_anchor_c11_sqrt` 使用 C11 full，validation macro-F1 接近阶段四最强单模型，是 anchor。

        Meta learner 采用 Logistic Regression，输入为 base learner 的 OOF 预测结果。训练时先在 train split 上做 5-fold OOF，避免 meta learner 直接学习 base learner 在训练样本上的过拟合输出；validation 集只用于比较 ensemble 方案。最终比较了 small-only stacking、small+anchor stacking、small probability average 和 probability average + anchor。结果显示，small-only stacking 的 validation macro-F1 = 0.7981，说明小模型互补性存在但不足以超过强单模型；简单概率平均 + anchor 的 macro-F1 = 0.8009，低于学习式 stacking；small + LGBM anchor stacking 的 validation macro-F1 = 0.8178，少数类 recall = 0.8151，相比阶段四最强单模型 0.8131 提升 0.0047。

        因此，最终封存模型确定为 `stack_small_plus_lgbm_anchor`，在最终 train+validation 重训和 test 评价阶段命名为 **Garbage Net**。该命名指代的不是单个 LightGBM，而是由多个传统机器学习模型与 LightGBM anchor 组成的 stacking 集成框架。
        """,
        """
        ## 11. 最终 test 集评价

        最终评价严格遵循“validation 选模、test 封存评价”的协议。模型结构、base learner 列表、meta learner 类型和参数均在 validation 阶段确定；test 集不参与任何特征选择、参数搜索或 ensemble 方案比较。最终训练时，train 与 validation 合并为 trainval，共 10420 张图像；在 trainval 上重新生成 5-fold OOF 预测训练 meta learner，然后各 base learner 使用完整 trainval 重训，最后对 1839 张 test 图像预测一次。

        Garbage Net 在 test 集上取得 accuracy = 0.8238、macro precision = 0.8213、macro recall = 0.8029、macro-F1 = 0.8097、weighted-F1 = 0.8228，少数类 recall 均值 = 0.7420。与最终单模型 LGBM C11 anchor 相比，anchor 的 test macro-F1 = 0.7950、accuracy = 0.8097、少数类 recall = 0.7231；Garbage Net 在 macro-F1 上提升约 0.0147，在 accuracy 上提升约 0.0141，说明 stacking 的提升在 test 上仍然成立，而不是只出现在 validation 上。

        各类别指标显示，clothes 表现最好，precision = 0.9085、recall = 0.9437、F1 = 0.9257；shoes 的 F1 = 0.8345，cardboard 的 F1 = 0.8401，glass 的 F1 = 0.8280，battery 的 F1 = 0.8230，biological 的 F1 = 0.8155，paper 的 F1 = 0.8139。相对较弱的类别包括 plastic、metal 和 trash，其中 plastic precision = 0.7686、recall = 0.7333、F1 = 0.7505；metal F1 = 0.7586；trash precision = 0.8542 但 recall 只有 0.6029，F1 = 0.7069。trash 的高 precision 和低 recall 表明模型在预测为 trash 时较谨慎，但漏掉了较多真实 trash 样本。

        validation 到 test 的表现存在正常回落：validation stacking macro-F1 = 0.8178，test macro-F1 = 0.8097，下降约 0.0081。该差距在图像分类任务中属于可接受范围，说明模型没有明显过拟合 validation。最终图表包括模型对比、Garbage Net 混淆矩阵和逐类别 precision/recall/F1，能够从整体和类别两个层面展示模型效果。
        """,
        """
        ## 12. 错误分析

        错误分析用于判断模型短板来自参数设置、特征表达还是数据本身的视觉歧义。最终错误对显示，最常见混淆是 plastic -> glass，共 24 次；glass -> plastic 共 15 次。这一对错误符合视觉直觉：塑料瓶、透明包装和玻璃瓶在颜色、透明度、边缘反光和局部纹理上较接近，手工颜色直方图和 HOG 边缘特征难以稳定区分材料本身。cardboard -> plastic 有 13 次，说明包装类图像中纸板与塑料外包装可能共享相似颜色和矩形轮廓。

        其他高频错误包括 glass -> shoes、paper -> shoes、plastic -> cardboard、plastic -> paper、plastic -> metal、shoes -> clothes、shoes -> paper、trash -> metal 等。这些错误并不完全是模型“分类能力不足”，也与图像采集状态有关。例如 shoes 和 clothes 都可能包含布料纹理；paper 和 cardboard 都属于纤维材料，颜色与纹理相近；trash 类本身定义更宽泛，可能包含金属、塑料、纸张等多种外观，因此容易被模型划入更具体的材料类别。

        从逐类别指标看，trash 是最终模型最主要短板。trash test recall = 0.6029，明显低于 battery 的 0.8230 和 biological 的 0.8000。其原因至少包括两点：第一，trash 样本只有 453 张，train 中仅 317 张，模型可学习的视觉模式较少；第二，trash 类语义更像“不能归入其他类别的杂项”，类内差异远大于 cardboard、paper、shoes 等相对稳定类别。对于这种类别，单纯继续调分类器参数的收益有限，更有效的方向可能是补充 trash 样本、进行类别内子类型标注、加入主体分割或使用深度特征增强语义表达。

        误分类样本网格展示了模型高置信度错误。高置信度错误尤其值得关注，因为它说明模型不仅判断错误，而且对错误类别非常确信。这类样本可以作为后续数据清洗和特征改进的重点对象：若图像主体不清晰或标签本身存在争议，应考虑人工复核；若标签无误，则说明当前手工特征没有捕捉到区分该样本所需的信息。
        """,
        """
        ## 13. 与已有传统机器学习和神经网络结果的对比

        本项目另外整理了若干传统机器学习文献方案的本地复刻结果，用作横向参照。复刻候选包括 Yang & Thung 2016 的 SIFT BoF + RBF SVM、Nguyen et al. 2026 的手工特征 + LightGBM/XGBoost、Mboli & Ogungbemi 2026 的 RGB/PCA + Random Forest/RBF SVM，以及 Sen et al. 2025 的 HOG+LBP 风格纹理 + Linear SVM。这些复刻只在 validation split 上评价，不参与核心模型选择，也不使用 test 集。

        传统文献候选中，Nguyen 风格的 handcrafted LightGBM 表现最好，validation macro-F1 = 0.7653，少数类 recall = 0.7932；handcrafted XGBoost 的 macro-F1 = 0.7014，少数类 recall = 0.8056；RGB/PCA + RBF SVM 的 macro-F1 = 0.5550；RGB/PCA + Random Forest 的 macro-F1 = 0.5229；HOG+Texture Linear SVM 的 macro-F1 = 0.4846；Yang & Thung 风格 SIFT BoF + RBF SVM 的 macro-F1 = 0.4240。相比之下，Garbage Net validation macro-F1 = 0.8178，说明本项目的分阶段特征融合、类别权重和 stacking 设计相较直接复刻文献候选有明显提升。

        与神经网络方法相比，Garbage Net 的定位不是替代高性能迁移学习或深度混合模型。已有研究表明，预训练 CNN、EfficientNet、DenseNet、ResNet 或深度特征 + 传统分类器在大规模图像分类中通常具有更高性能上限，尤其能够自动学习复杂语义特征。但深度模型也带来训练成本、可解释性、环境依赖和调参复杂度问题。本项目选择传统机器学习路线，是为了在可控计算成本下完整展示“图像 -> 手工特征 -> 特征融合 -> 模型筛选 -> 调参消融 -> 集成学习 -> 错误分析”的全过程。

        因此，Garbage Net 更适合作为一个强传统机器学习基线。其价值在于：特征来源清楚，能够解释 HOG、Color、Texture、BoF 等特征的贡献；模型选择过程透明，能够说明 LightGBM 为什么成为主模型；类别不平衡处理有可比较结果，能够说明 `sqrt_balanced` 的作用；最终 stacking 的增益也能通过 validation 和 test 两个阶段验证。
        """,
        """
        ## 14. 总结与反思

        本项目最终形成的 Garbage Net 是一个基于手工特征与传统机器学习的 stacking 集成模型。实验结果表明，传统方法在经过系统特征工程和模型组合后，可以在 10 类垃圾图像分类任务上取得较强表现。最终 test macro-F1 = 0.8097，accuracy = 0.8238，weighted-F1 = 0.8228，相比单模型 LightGBM anchor 的 test macro-F1 = 0.7950 有进一步提升。

        从建模过程看，最关键的结论包括五点。第一，HOG 是最强单特征，调参后 HOG+KNN 的 validation macro-F1 达到 0.5612，是后续融合的自然起点。第二，Color 和 Texture 对少数类 recall 具有重要价值，Color+KNN 的少数类 recall 达到 0.5620，因此不能只根据单一 macro-F1 排名删除这些特征。第三，LightGBM 是最适合本项目高维混合手工特征的单模型，阶段三 C11 full + LightGBM 的 validation macro-F1 达到 0.8096，显著高于 Random Forest、Linear SVM、Logistic Regression、KNN 和 Decision Tree。第四，特征消融证明 BoF 在融合中具有真实贡献，去掉 BoF 后 macro-F1 从 0.8096 降至 0.8020，少数类 recall 也从 0.8158 降至 0.7961。第五，`sqrt_balanced` 是较合适的不平衡处理方式，使 C11 LightGBM validation macro-F1 提升到 0.8131。

        Stacking 的作用也较明确。小模型本身并不一定强，例如 KNN compact 的 validation macro-F1 仅为 0.5492，Linear SVM 为 0.6139，但它们提供了与 LightGBM 不同的错误模式和输出视角。当这些模型与 LGBM anchor 一起进入 meta learner 后，validation macro-F1 提升到 0.8178，test macro-F1 也达到 0.8097。该结果说明，异质模型集成在传统特征体系中仍具有实际意义。

        模型短板主要集中在 trash 类和 plastic/glass 等易混淆类别。后续改进可从三方面展开：一是补充 trash 等少数类样本，或进行更有针对性的数据增强；二是改进主体区域提取，减少背景对颜色和纹理统计的干扰；三是在保持传统模型可解释性的前提下，引入深度特征作为额外对照，进一步比较手工特征、深度特征和混合特征的边界。
        """,
        """
        ## 附录 A：可复现性说明

        本报告文件使用项目内固定的数据划分、特征缓存和模型报告文件生成。所有路径均采用相对路径，不依赖本地绝对路径。核心输入包括 `data/processed/splits/split_summary.csv`、`data/processed/feature_reports/feature_extraction_report.csv`、`stage2` 到 `stage4` 的模型报告，以及最终 test 评价文件 `final_test_evaluation_summary.json`、`final_model_results.csv`、`final_per_class_metrics.csv` 和 `final_error_pairs.csv`。

        可复现性主要体现在四个方面。第一，数据划分固定，所有模型使用同一套 train/validation/test split，避免不同实验因样本划分不同而不可比较。第二，特征提取结果已经缓存为 `.npz` 和报告文件，notebook 不重新提取特征，而是读取固定特征矩阵与汇总结果，保证报告生成稳定。第三，模型筛选、调参、消融和 stacking 的中间结果均保存在 CSV/JSON 文件中，notebook 中的表格和图均由这些文件自动生成。第四，最终 test 评价只对应封存后的 Garbage Net，不在 notebook 中再次进行面向 test 的参数调整。

        PDF 导出时建议先完整运行 notebook，确认所有代码单元都有输出，再执行 `jupyter nbconvert --to pdf Garbage_Net.ipynb` 或在 Jupyter/Lab 界面中导出。若环境缺少 LaTeX，也可以先导出 HTML，再通过浏览器打印为 PDF。
        """,
    ]

    additional_markdown_details = [
        """
        摘要中的指标均来自最终封存模型在 test split 上的一次性评价，而不是 validation 阶段的选择结果。该区分非常重要：validation macro-F1 = 0.8178 用于选择 Garbage Net，test macro-F1 = 0.8097 用于报告最终泛化表现。两者相差约 0.0081，说明最终模型在未参与调参的数据上仍保持稳定。摘要同时保留少数类 recall 均值，是因为垃圾分类在实际应用中不能只追求整体正确率；若 trash、battery、biological 等类别长期被漏判，即使 accuracy 较高，也不能说明模型真正可靠。
        """,
        """
        每个阶段的代码输出都承担不同作用。数据阶段的输出用于确认样本量和划分是否正确；特征阶段的输出用于确认维度、缺失值和缓存状态；阶段二输出用于确定后续融合的起点；阶段三热力图用于比较模型族与特征组合的交互；阶段四输出用于解释调参失败、消融收益和类别权重收益；最终评价输出用于证明模型选择在 test 上仍成立。这样的结构使 notebook 不只是结果展示，而是把实验决策链条完整呈现出来。
        """,
        """
        文献综述还限定了本项目的边界。深度学习和混合深度特征方法在已有研究中具有更高性能上限，但它们往往需要更复杂的训练环境、更多调参和更强硬件。本项目的研究重点是传统机器学习条件下的完整建模流程，因此不会把深度学习作为主体实验。非线性核 SVM 同样不作为主线：其历史基准已经存在，本地复刻表现也不强，而广筛阶段需要在大量特征组合上重复训练，核方法的计算结构与本项目规模不匹配。
        """,
        """
        下面代码输出中的柱状图和堆叠图对应两个检查点。第一，类别分布柱状图用于暴露不平衡程度，其中 trash 明显低于其他类别；第二，split 堆叠图用于检查分层抽样是否保持各类别比例。若某一类别在 validation 或 test 中样本过少，则 macro-F1、recall 和混淆矩阵都可能不稳定。本项目每类 validation/test 样本均保持可评价规模，例如 trash 在 validation/test 中各 68 张，虽然仍是少数类，但足以进行基本误差分析。
        """,
        """
        特征维度也反映了后续模型选择的难度。F3 HOG 1764 维和 F1 Color 1039 维占据主要维度，二者拼接后已经达到 2803 维；加入 Texture 后为 2833 维，加入 GIST、Shape、BoF 后最终达到 3165 维。高维连续特征对 KNN 和核 SVM 的计算较不友好，但对 LightGBM 这类基于树的 boosting 模型较合适，因为树模型可以通过分裂规则自动选择有效特征区间。低维特征如 F2、F4、F5 虽然单独信息有限，却能在融合和 stacking 中提供补充视角。
        """,
        """
        从输出表可以看出，固定参数阶段并没有出现某一个特征全面领先所有指标的情况。HOG+KNN 的 macro-F1 最高，但少数类 recall 只有 0.4344；Texture+Logistic Regression 的 macro-F1 低于 HOG，但少数类 recall 达到 0.5555；Color+KNN 的少数类 recall 也达到 0.5470。这种差异说明，若只按 macro-F1 选择特征，可能会错过对少数类有帮助的颜色和纹理信息。因此后续融合不是简单取 top-1 特征，而是保留多种互补特征。
        """,
        """
        调参输出还说明，模型参数会改变特征表现。HOG+KNN 从欧氏距离、k=5 调整到 manhattan 距离、k=9、distance 权重后，macro-F1 提升到 0.5612；Color+KNN 使用 MinMaxScaler、cosine 距离、k=11 后，更适合颜色直方图这类非负分布特征；Texture+KNN 使用 RobustScaler 后，可以降低纹理统计量中异常值对距离计算的影响。这些结果表明，传统机器学习中的 preprocessing 与模型参数同样重要，不能只讨论分类器名称。
        """,
        """
        阶段三热力图用于观察两个维度的规律。沿模型维度看，LightGBM 在大多数强特征组合上领先，说明它对混合特征的非线性建模能力更强；沿特征维度看，从 F3 到 F3+F1，再到 F3+F1+F2，macro-F1 持续上升，说明颜色和纹理不是冗余信息。C11 full 相比 HOG anchor 的 macro-F1 增益约 0.1728，少数类 recall 增益约 0.2228，是整个实验中最关键的性能跃迁之一。
        """,
        """
        精调阶段的输出需要结合两个指标解读。LGBM_A_C11_full 的 macro-F1 低于阶段三，但少数类 recall 升高到 0.8272；LGBM_B_no_shape 的少数类 recall 达到 0.8302，但 macro-F1 进一步下降到 0.8000。这说明 sample weight 和参数随机搜索会改变模型偏好：更强调少数类时，部分类别召回提高，但其他类别 precision 或 F1 可能下降。由于主指标是 macro-F1，不能仅因为少数类 recall 提高就替换主模型。
        """,
        """
        消融表中的数值为最终特征选择提供了直接证据。no BoF 的下降幅度大于 no Shape，说明 BoF 的边际贡献更重要；lowdim Color+Texture+GIST 的 macro-F1 只有 0.7845，但少数类 recall 达到 0.8104，说明低维组合虽然整体弱于 C11，却对少数类有较好覆盖。类别权重表则显示 `sqrt_balanced` 比 `balanced` 更稳：它没有继续提高少数类 recall，但把 macro-F1 从 0.8096 提高到 0.8131，说明更温和的权重能减少过度补偿。
        """,
        """
        Stacking 输出中的一个关键现象是，简单平均并没有充分利用 base learner。small probability average 的 validation macro-F1 = 0.7686，probability average + anchor 为 0.8009，均低于学习式 stacking。原因在于不同 base learner 的可靠性不同，简单平均会给弱模型过高权重；Logistic Regression meta learner 可以根据 OOF 预测自动学习各类别和各模型的相对可信度。因此，Garbage Net 的改进来自“有监督地融合预测”，而不是机械平均。
        """,
        """
        最终 test 表还可以验证 ensemble 的实际收益。final small-only stacking 的 test macro-F1 = 0.8036，低于 Garbage Net 的 0.8097，但高于很多单个小模型，说明小模型组合已经有一定互补性；加入 LGBM anchor 后，模型同时获得强单模型的稳定性和小模型的补充信息。weighted-F1 = 0.8228 高于 macro-F1 = 0.8097，说明样本较多类别表现更好，而 macro-F1 更严格地反映了少数类和困难类的影响。
        """,
        """
        错误分析阶段的图表不用于重新选择模型，而用于解释模型行为。若在看到 test 错误后继续调参，会破坏 test 的封存性质。因此，错误分析只用于提出后续改进方向。plastic/glass 的互相混淆提示需要更强的材质表达；trash/metal 的混淆提示 trash 类语义边界不清；paper/shoes、shoes/clothes 等错误提示背景、纹理和主体形态会影响手工特征。后续改进应重新回到 train/validation 协议中进行，而不是直接针对 test 错误优化。
        """,
        """
        文献对比部分的柱状图应理解为方法定位，而不是严格公平的排行榜。不同文献候选对应的特征维度、类别空间、参数完整性和本地映射方式并不完全相同，因此该表只说明本项目路线相较这些复刻候选更适合当前数据和实验目标。尤其是 Nguyen 风格 handcrafted LightGBM 已经达到 0.7653，说明“多手工特征 + boosting”方向本身有效；Garbage Net 的进一步提升来自更细的特征融合、权重选择和 stacking，而不是单纯更换模型名称。
        """,
        """
        总体来看，本项目的结果支持两个判断。第一，传统机器学习并非只能作为低性能 baseline；在特征工程充分、模型筛选系统、类别不平衡处理合理的情况下，传统方法仍能形成较强分类器。第二，传统方法的上限高度依赖特征表达，如果输入特征没有捕捉到材料语义或主体区域，后续模型很难弥补。因此，Garbage Net 的意义在于给出了一条结构清晰、结果可复查的传统建模路线，同时也明确指出了未来引入更强视觉表征的必要性。
        """,
        """
        附录中的 artifact 检查用于保证报告不是手工摘录结果。若某个关键 CSV/JSON 缺失，notebook 对应表格会无法生成，从而暴露复现问题。依赖版本记录也有必要保留，因为 LightGBM、scikit-learn、numpy 等库版本变化可能影响训练速度、默认参数或数值细节。最终交付 PDF 前，应保留已执行输出，这样评阅者可以直接看到每个代码单元的自动结果，而不需要重新运行耗时实验。
        """,
    ]

    if len(additional_markdown_details) != len(expanded_markdown_sources):
        raise RuntimeError("Additional markdown detail count mismatch")

    expanded_markdown_sources = [
        clean(base).rstrip() + "\n\n" + clean(extra).strip()
        for base, extra in zip(expanded_markdown_sources, additional_markdown_details)
    ]

    final_markdown_details = [
        """
        需要特别说明的是，Garbage Net 的命名用于报告最终模型结构，而不是暗示其为神经网络。这里的 “Net” 指多个传统学习器通过 stacking 形成的预测网络：底层模型负责从不同特征视角产生类别证据，顶层 meta learner 负责综合这些证据。该命名能够概括模型的集成结构，同时保留传统机器学习路线的可解释性。
        """,
        """
        评价协议贯穿所有阶段。所有中间表格中的 `validation_macro_f1` 都用于选择方案，所有最终表格中的 `test_macro_f1` 才用于报告最终泛化。少数类 recall 均值固定取 trash、battery、biological 三类，是因为这三类在样本量、视觉稳定性和实际错误成本上都更需要关注。若某方案 macro-F1 提升但少数类 recall 大幅下降，则不会被直接采纳。
        """,
        """
        文献综述对本项目的实际作用体现在特征和模型两个层面。特征层面，已有研究支持颜色、纹理、形状、局部关键点和全局结构共同建模，因此 F1-F6 被设计为互补特征组。模型层面，已有研究显示 boosting 和传统分类器仍有价值，但非线性核 SVM 已被早期垃圾分类工作充分尝试，因此本项目把它作为复刻对照而非主线创新点。
        """,
        """
        数据划分采用固定随机种子和分层抽样，目的是把实验差异尽量限制在特征、模型和参数本身。若每个阶段重新划分数据，validation 波动会掩盖真实模型差异，也会使单特征、融合、消融和 stacking 的结果难以横向比较。固定 split 因此是整个实验协议的基础。
        """,
        """
        特征提取阶段没有进行端到端学习，因此每一组特征都承担明确语义。颜色特征适合区分 biological、glass、plastic 等颜色分布明显的类别；纹理特征适合纸张、纸板、布料和金属表面；HOG 负责主体边缘和轮廓；BoF 关注局部关键点；GIST 关注整体布局。后续模型性能差异本质上反映了这些语义信息被利用的程度。
        """,
        """
        固定参数 baseline 的价值在于建立“朴素起点”。如果某个特征在简单模型和固定参数下已经完全无效，则后续继续投入大量搜索通常不划算；如果某个特征在部分指标上表现突出，即使总体排名不是第一，也应保留到融合阶段验证。阶段二结果正是以这种方式筛出了 HOG、Color、Texture、GIST 和 BoF。
        """,
        """
        调参阶段没有追求最大搜索空间，而是采用受控搜索。这样做符合本项目目标：重点展示合理决策，而不是让随机搜索主导结果。调参输出中保留 `search_strategy`、`cv_macro_f1_mean`、`validation_macro_f1` 和 `best_params`，可以追踪每个候选为何被保留或舍弃。
        """,
        """
        阶段三中的 98 组实验构成了主模型选择的核心证据。若只比较最终 C11 组合，无法判断 Color、Texture、BoF、Shape 是否真的贡献有效信息；若只比较 LightGBM，无法说明其他模型族为何被排除。热力图把这两个维度同时展开，使模型选择具有可解释依据。
        """,
        """
        强模型精调阶段体现了实验中的负结果价值。虽然调参没有刷新最高 macro-F1，但它揭示了参数搜索可能带来的指标权衡：某些方案提高少数类 recall，却降低整体 macro-F1。这种负结果避免了继续在同一方向投入计算，也为后续转向消融和类别权重提供了理由。
        """,
        """
        消融实验是判断特征贡献的关键步骤。与单特征实验不同，消融关注的是某组特征在强模型和融合环境中的边际价值。BoF 单独不强但消融后下降明显，正说明特征价值不能只由单特征结果决定；Shape 单独弱但删除后略降，说明低成本补充特征在全量模型中仍可能有用。
        """,
        """
        Stacking 的设计遵循“强 anchor + 异质补充”的原则。若只使用小模型，模型上限不足；若只使用强 LightGBM，则缺少其他学习偏好的补充。Garbage Net 同时保留 C11 anchor、紧凑 LightGBM、线性模型、森林模型和邻域模型，使 meta learner 能在类别层面学习不同 base learner 的可信度。
        """,
        """
        最终 test 输出中的混淆矩阵、per-class 指标和错误对表互相补充。总体指标说明模型强弱，per-class 指标说明哪些类别表现稳定，混淆矩阵显示错误分布，错误对表则直接指出最值得分析的类别对。四类输出合在一起，才能避免只用一个 macro-F1 概括所有问题。
        """,
        """
        错误样本展示尤其适合检查“高置信度错误”。若模型以 0.98 以上置信度把 plastic 判为 glass，说明当前特征在该样本上给出了非常强但错误的证据。这类错误往往不是简单调阈值能解决，而需要更好的材料特征、更可靠的主体区域或更多相似样本参与训练。
        """,
        """
        文献对比还说明，单纯复刻已有方法并不等于完成本数据集上的最优建模。相同模型族在不同数据、类别数量、图像质量和参数设置下可能表现差异很大。因此，本项目把文献方法作为启发和参照，再通过本地 validation 协议重新验证，而不是直接照搬文献结论。
        """,
        """
        总结部分的重点是方法论：传统机器学习项目同样需要系统实验设计。若没有单特征基线，就无法解释融合来源；若没有消融，就无法证明全量特征必要性；若没有类别权重比较，就无法说明少数类处理；若没有 test 封存，就无法证明最终结果可靠。Garbage Net 的价值建立在这些步骤共同支撑之上。
        """,
        """
        附录中的依赖版本和 artifact 列表也方便后续转 PDF 或复查。如果 notebook 在其他环境中重新运行，最可能出现差异的是字体渲染、图库版本和 LightGBM/sklearn 的显示格式；核心结果由于来自固定报告文件，应保持一致。这样既满足作业展示需求，也保留了复现实验的入口。
        """,
    ]

    if len(final_markdown_details) != len(expanded_markdown_sources):
        raise RuntimeError("Final markdown detail count mismatch")

    expanded_markdown_sources = [
        clean(base).rstrip() + "\n\n" + clean(extra).strip()
        for base, extra in zip(expanded_markdown_sources, final_markdown_details)
    ]

    md_idx = 0
    for cell in cells:
        if cell.cell_type == "markdown":
            cell.source = clean(expanded_markdown_sources[md_idx])
            md_idx += 1
    if md_idx != len(expanded_markdown_sources):
        raise RuntimeError(f"Markdown replacement mismatch: {md_idx} != {len(expanded_markdown_sources)}")

    nb.cells = cells
    return nb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-execute", action="store_true", help="Only write the notebook skeleton.")
    args = parser.parse_args()

    nb = build_notebook()
    nbf.write(nb, NOTEBOOK_PATH)

    if not args.no_execute:
        client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": "."}})
        client.execute()
        nbf.write(nb, NOTEBOOK_PATH)

    print(f"Wrote {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
