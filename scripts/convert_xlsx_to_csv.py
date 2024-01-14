import pandas as pd

combined_file: str = (
    "/home/mbrzus/programming/dcm_train_data/combined/combined_predicthd_data_Jan9.xlsx"
)
df = pd.read_excel(combined_file)
df.to_csv(combined_file.replace(".xlsx", ".csv"), index=False)
