import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
df = pd.read_csv("apollo_data.csv", index_col = 0)
df.head()

df.info()

df.describe()

# Convert relevant columns to 'category' data type
categorical_cols = ["sex", "smoker", "region"]
for col in categorical_cols:
    df[col] = df[col].astype("category")

# Apply one-hot encoding for categorical features (drop_first avoids multicollinearity)
df_encoded = pd.get_dummies(df, drop_first=True)

# Show the cleaned and preprocessed DataFrame
df_encoded.head()

#Question 1: Which variables are significant in predicting the reason for hospitalization for different regions?

'''
Approach
We’ll approach this by:

Performing ANOVA and Chi-square tests to identify if distributions of key variables differ significantly by region.
Testing continuous variables like age, viral load, and severity level using ANOVA.
Testing categorical variables like sex and smoker using Chi-Square Test of Independence.
'''
# Step 1: ANOVA – Do continuous variables vary across regions?
# For each continuous variable, perform one-way ANOVA across regions
anova_age = stats.f_oneway(
    df[df["region"] == "northeast"]["age"],
    df[df["region"] == "southeast"]["age"],
    df[df["region"] == "southwest"]["age"],
    df[df["region"] == "northwest"]["age"]
)

anova_viral = stats.f_oneway(
    df[df["region"] == "northeast"]["viral load"],
    df[df["region"] == "southeast"]["viral load"],
    df[df["region"] == "southwest"]["viral load"],
    df[df["region"] == "northwest"]["viral load"]
)

anova_severity = stats.f_oneway(
    df[df["region"] == "northeast"]["severity level"],
    df[df["region"] == "southeast"]["severity level"],
    df[df["region"] == "southwest"]["severity level"],
    df[df["region"] == "northwest"]["severity level"]
)

# Print results
anova_age, anova_viral, anova_severity

# For each continuous variable, perform one-way ANOVA across regions
anova_age = stats.f_oneway(
    df[df["region"] == "northeast"]["age"],
    df[df["region"] == "southeast"]["age"],
    df[df["region"] == "southwest"]["age"],
    df[df["region"] == "northwest"]["age"]
)

anova_viral = stats.f_oneway(
    df[df["region"] == "northeast"]["viral load"],
    df[df["region"] == "southeast"]["viral load"],
    df[df["region"] == "southwest"]["viral load"],
    df[df["region"] == "northwest"]["viral load"]
)

anova_severity = stats.f_oneway(
    df[df["region"] == "northeast"]["severity level"],
    df[df["region"] == "southeast"]["severity level"],
    df[df["region"] == "southwest"]["severity level"],
    df[df["region"] == "northwest"]["severity level"]
)

# Print results
anova_age, anova_viral, anova_severity


'''
ANOVA Results Summary
We tested whether continuous variables (age, viral load, severity level) vary significantly across different regions using one-way ANOVA.

Age:
F(3, 1334) = 0.08, p = 0.97 ❌
→ No significant difference in average age across regions.

Viral Load:
F(3, 1334) = 39.47, p < 0.001 ✅
→ Highly significant difference in viral load between regions. This suggests that the severity of viral exposure varies geographically.

Severity Level:
F(3, 1334) = 0.77, p = 0.54 ❌
→ No statistically significant difference in severity level across regions.

📌 Insight: Among the continuous predictors, only viral load shows meaningful variation across regions, which may reflect differing infection rates or testing/reporting practices by location.
'''

# Step 2: Chi-Square Test – Are sex and smoker status independent of region?

# Cross-tabulation and Chi-Square for 'sex' vs. 'region'
contingency_sex = pd.crosstab(df["region"], df["sex"])
chi2_sex = stats.chi2_contingency(contingency_sex)

# Cross-tabulation and Chi-Square for 'smoker' vs. 'region'
contingency_smoker = pd.crosstab(df["region"], df["smoker"])
chi2_smoker = stats.chi2_contingency(contingency_smoker)

# Show test statistics and p-values
chi2_sex[0:2], chi2_smoker[0:2]

'''
Chi-Square Test Results
We assessed whether the distribution of categorical variables (sex, smoker) is independent of the region:

Sex vs Region
χ² = 0.43, p = 0.93 ❌
→ No relationship between gender distribution and region. Gender is evenly spread geographically.

Smoker vs Region
χ² = 7.34, p = 0.061 ❌ (borderline)
→ While not statistically significant at p < 0.05, there is a weak regional trend in smoking behavior (marginal significance).

📌 Insight: Neither sex nor smoking status vary significantly by region, although smoking status comes close to the significance threshold. This may warrant deeper exploration in future studies.

From our statistical analysis:

✅ Viral Load is the only variable that shows a significant difference across regions.
❌ Age, severity level, sex, and smoking status do not vary significantly by region.
This suggests that while reasons for hospitalization may be influenced by local viral exposure levels, other demographic and behavioral factors are evenly distributed across regions.


'''
#How well some variables like viral load, smoking, and severity level describe the hospitalization charges?
'''
Apollo is interested in understanding whether factors like viral load, smoking, and severity level can reliably predict hospitalization charges. To answer this, we use linear regression, which helps quantify how much each variable contributes to cost differences.

We'll also account for other potential confounding variables, such as:

Age
Sex
Region
to ensure a robust and interpretable model.


'''
#Linear Regression Model
# Linear regression model to predict hospitalization charges

# Define target variable (y) and features (X)
y = df_encoded["hospitalization charges"]
X = df_encoded.drop(columns=["hospitalization charges"])

# Add intercept to the model
X = sm.add_constant(X)

# Fit the OLS regression model
model = sm.OLS(y, X).fit()

# Show summary of the model
model.summary()

'''
Linear Regression Test Results
We used a multiple linear regression model to quantify how well various predictors explain hospitalization charges. The model includes:

Biological factors: viral load, severity level
Behavioral: smoker
Demographic: age, sex
Geographic: region (dummy encoded)
🔧 Model Fit & Significance
R-squared = 0.751 → Model explains ~75.1% of variance in charges ✅
F-statistic = 500.9, p < 0.001 ✅ → Model is statistically significant
n = 1338 observations
'''
'''
Insights and Recommendations
Based on the statistical analyses and modeling conducted, we outline the following key insights and strategic recommendations for Apollo Hospitals:

Key Insights
Viral Load Varies by Region

Viral load was the only continuous variable showing statistically significant differences across regions.
This may reflect varying levels of infection exposure or reporting between geographical areas.
Smoking Has the Largest Impact on Cost

Smoking is the most influential variable in predicting hospitalization charges.
Smokers incur, on average, nearly 60,000 units more in charges than non-smokers.
Biological Severity Drives Cost

Both viral load and severity level significantly increase hospitalization charges.
This aligns with clinical expectations: sicker patients cost more to treat.
Demographics Have Limited Cost Impact

Age slightly increases cost (roughly +640 per year).
Sex does not significantly affect hospitalization charges.
Regional Differences in Cost

Patients from southeast and southwest regions tend to have lower costs than those in the northeast.
This may be due to hospital infrastructure, local pricing, or clinical practice variation.
Recommendations
Target Smoking Cessation Programs

Prioritize public health initiatives and awareness campaigns to reduce smoking rates.
Investing in prevention could reduce hospitalization costs significantly over time.
Resource Allocation Based on Viral Load Hotspots

Monitor regions with high average viral loads for early intervention and preparedness.
Focus testing, isolation, and outreach in these areas during peak outbreaks.
Severity-Based Risk Adjustment

Consider incorporating severity level into triage or billing strategies.
Use it for early identification of high-cost cases and personalized care plans.
Audit Regional Cost Variability

Investigate why some regions have significantly lower hospitalization costs.
Explore if efficiency practices from these areas can be replicated across the network.
Incorporate Predictive Models into Hospital Operations

Embed models like this into operational systems to estimate likely cost at intake.
Enables proactive resource planning and financial forecasting.
'''