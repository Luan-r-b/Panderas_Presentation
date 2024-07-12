import pandas as pd
import pandera as pa
from pandera import DataFrameModel, Field, check_types, infer_schema, Check, check_input, Column, DataFrameSchema
from pandera.typing import DataFrame, Series
import pandera.extensions as extensions
from scipy.stats import ttest_ind


class MedCost(DataFrameModel):
    Id: Series[int] = Field(gt=0, nullable=False, unique=True)
    age: Series[int] = Field(ge=0, nullable=False)
    sex: Series[str] = Field(isin=["female", "male"])
    bmi: Series[float]
    children: Series[int] = Field(ge=0)
    smoker: Series[str] = Field(isin=["yes", "no"])
    region: Series[str] = Field(isin=["southwest", "southeast", "northwest", "northeast"])
    charges: Series[float] = Field(ge=0)

    @pa.check("bmi", name="check_bmi")
    def check_bmi(cls, bmi : Series[float]) -> Series[bool]:
        return bmi < 100

    @pa.dataframe_check
    def validate_charges(csl, df: pd.DataFrame) -> pd.Series:
        min_charges = (df['age'] + df['bmi']) * 15
        max_charges = (df['age'] + df['bmi']) * 850
        return df['charges'].between(min_charges, max_charges)

class FemaleMedCost(MedCost):
    charges: Series[float] = Field(ge=0)

    class Config:
        drop_invalid_rows = True

    @pa.dataframe_check
    def validate_charges(csl, df: pd.DataFrame) -> pd.Series:
        max_charges = df["children"] * 1000 + 5000
        return df["charges"] <= max_charges
    

class SmokerMedCost(MedCost):

    @pa.dataframe_check
    def validate_charges(cls, df: pd.DataFrame) -> pd.Series:
        return True

    @pa.dataframe_check(name="smoker_vs_non_smoker_charges")
    def validate_smoker_charges(cls, df: pd.DataFrame) -> Series[bool]:
        smokers = df[df["smoker"] == "yes"]["charges"]
        non_smokers = df[df["smoker"] == "no"]["charges"]
        
        t_stat, p_value = ttest_ind(smokers, non_smokers, equal_var=False, alternative="greater")

        # Reject the null hypothesis if p-value < 0.05, indicating smokers have higher charges
        return pd.Series([p_value < 0.05])


