import streamlit as st
from src.streamlit_utils import load_tests


def page_hypotheses_validation_body():
    st.title("Hypotheses Validation")
    st.markdown(
        """
        On this page, we check three core assumptions ("hypotheses") about our data and model:

        1. **Do powdery-mildew leaves show more variation in pixel brightness than healthy leaves?**
        2. **Is the average brightness of mildew-infected leaves different from healthy ones?**
        3. **Does our model recall infected leaves significantly better than random guessing?**
        """
    )

    st.info(
        """
        **Note:** Hypothesis 4 (that lowering the learning rate and using EarlyStopping
        yields smoother convergence and higher validation accuracy) was already
        evaluated in **Notebook 03: Modelling & Evaluating** and in the
        Training Dashboard. This page focuses on the remaining statistical tests
        (Hypotheses 1–3).
        """
    )

    tests = load_tests()

    # --- Variance Hypothesis ---
    st.header("1. Pixel Variance Hypothesis")
    st.markdown(
        """
        **What we tested:**
        Whether the tiny spots and texture changes in mildew leaves create **more variation**
        in pixel brightness than on healthy leaves.

        **How we tested it:**
        We ran a statistical test (Welch’s t-test) on the per-image brightness variances.

        **Result:**
        """
    )
    t_var = tests["t_var"]
    p_var = tests["p_var"]
    st.write(f"- **t-statistic:** {t_var:.2f} **p-value:** {p_var:.4f}")
    if p_var < 0.05:
        st.success(
            "✅ The difference is **statistically significant** (p < 0.05).\n\n"
            "This means mildew leaves truly do show more pixel-to-pixel variation than healthy leaves."
        )
    else:
        st.warning(
            "⚠️ The difference is **not** significant (p ≥ 0.05).\n\n"
            "We cannot conclude that variance differs between the two classes."
        )

    st.markdown("---")

    # --- Mean Intensity Hypothesis ---
    st.header("2. Pixel Mean Intensity Hypothesis")
    st.markdown(
        """
        **What we tested:**
        If the **average brightness** of mildew-infected leaves differs from that of healthy leaves.

        **How we tested it:**
        A two-sample t-test comparing the mean pixel intensities of each class.

        **Result:**
        """
    )
    t_mean = tests["t_mean"]
    p_mean = tests["p_mean"]
    st.write(f"- **t-statistic:** {t_mean:.2f} **p-value:** {p_mean:.4f}")
    if p_mean < 0.05:
        st.success(
            "✅ The mean brightness difference is **statistically significant** (p < 0.05).\n\n"
            "In practice, this means mildew leaves are detectably lighter or darker on average than healthy ones."
        )
    else:
        st.warning(
            "⚠️ The mean difference is **not** significant (p ≥ 0.05).\n\n"
            "We cannot be sure that average brightness differs by class."
        )

    st.markdown("---")

    # --- Recall Hypothesis ---
    st.header("3. Model Recall Hypothesis")
    st.markdown(
        """
        **What we tested:**
        Whether our model’s ability to correctly identify mildew images (**recall**) beats a **50% random baseline**.

        **How we tested it:**
        We treated each mildew test image as a “correct” (1) or “incorrect” (0) prediction
        and ran a one-sample t-test against the value 0.5 (random guessing).

        **Result:**
        """
    )
    t_rec = tests["t_rec"]
    p_rec = tests["p_rec"]
    st.write(f"- **t-statistic:** {t_rec:.2f} **p-value:** {p_rec:.4f}")
    if p_rec < 0.05:
        st.success(
            "✅ Model recall is **significantly above** 50% (p < 0.05).\n\n"
            "This confirms our CNN is reliably better than chance at finding infected leaves."
        )
    else:
        st.warning(
            "⚠️ Model recall is **not** significantly above 50% (p ≥ 0.05).\n\n"
            "This suggests the model may not outperform random guessing on recall."
        )

    st.markdown("---")
    st.info(
        "These statistical tests give us confidence that the key pixel features and the model’s performance "
        "are not due to random chance, supporting our business case for automated mildew detection."
    )
