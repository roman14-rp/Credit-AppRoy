import numpy as np
import pandas as pd

KOLOM_NUMERIK = ["LogTotalIncome", "LogLoanAmount", "Loan_Amount_Term",
                 "Credit_History", "Rasio_Cicilan", "Ada_Pendamping"]
KOLOM_KATEGORIK = ["Married", "Education", "Property_Area"]
KOLOM_MODEL = KOLOM_NUMERIK + KOLOM_KATEGORIK


def buat_fitur(df):
    """Ubah data mentah (kolom asli dataset) menjadi fitur yang dipakai model."""
    d = pd.DataFrame(index=df.index)
    pemohon = pd.to_numeric(df["ApplicantIncome"], errors="coerce")
    pendamping = pd.to_numeric(df["CoapplicantIncome"], errors="coerce").fillna(0)
    pinjaman = pd.to_numeric(df["LoanAmount"], errors="coerce")      # dalam ribuan
    tenor = pd.to_numeric(df["Loan_Amount_Term"], errors="coerce")   # dalam bulan

    total = pemohon + pendamping
    cicilan = (pinjaman * 1000) / tenor                              # cicilan pokok per bulan

    d["LogTotalIncome"] = np.log1p(total)                            # log untuk meredam outlier
    d["LogLoanAmount"] = np.log1p(pinjaman)
    d["Loan_Amount_Term"] = tenor
    d["Credit_History"] = pd.to_numeric(df["Credit_History"], errors="coerce")
    d["Rasio_Cicilan"] = (cicilan / total.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    d["Ada_Pendamping"] = (pendamping > 0).astype(int)
    for kolom in KOLOM_KATEGORIK:
        d[kolom] = df[kolom]
    return d[KOLOM_MODEL]
