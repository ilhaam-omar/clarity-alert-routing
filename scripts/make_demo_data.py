from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo_data"
HOSP = DEMO / "hosp"
ICU = DEMO / "icu"
HOSP.mkdir(parents=True, exist_ok=True)
ICU.mkdir(parents=True, exist_ok=True)

pd.DataFrame(
    [
        (1001, 2001), (1001, 2005), (1002, 2002), (1003, 2003), (1004, 2004),
    ], columns=["subject_id", "hadm_id"]
).to_csv(DEMO / "kidney_transplant_cohort.csv", index=False)

pd.DataFrame(
    [
        (1001, "F", 46, 2020),
        (1002, "M", 66, 2020),
        (1003, "M", 56, 2020),
        (1004, "F", 41, 2020),
    ], columns=["subject_id", "gender", "anchor_age", "anchor_year"]
).to_csv(HOSP / "patients.csv", index=False)

pd.DataFrame(
    [
        (1003, 1999, "2022-01-01 06:00:00", "2022-01-12 12:00:00"),
        (1001, 2001, "2024-01-01 07:00:00", "2024-01-10 12:00:00"),
        (1002, 2002, "2024-02-01 07:00:00", "2024-02-05 12:00:00"),
        (1003, 2003, "2024-03-01 07:00:00", "2024-03-05 12:00:00"),
        (1004, 2004, "2024-04-01 07:00:00", "2024-04-05 12:00:00"),
        (1001, 2005, "2024-04-15 07:00:00", "2024-04-18 12:00:00"),
    ], columns=["subject_id", "hadm_id", "admittime", "dischtime"]
).to_csv(HOSP / "admissions.csv", index=False)

prescriptions = [
    # Incident transplant admission: DDI repeat episode plus steroid transition.
    (1001,2001,"P1","Tacrolimus","2024-01-01 08:00:00","2024-01-10 08:00:00","PO",2,"mg",2),
    (1001,2001,"P2","Fluconazole","2024-01-01 12:00:00","2024-01-05 12:00:00","PO",200,"mg",1),
    (1001,2001,"P3","Fluconazole","2024-01-02 12:00:00","2024-01-05 12:00:00","PO",200,"mg",1),
    (1001,2001,"P4","Methylprednisolone","2024-01-01 09:00:00","2024-01-03 09:00:00","IV",100,"mg",2),
    (1001,2001,"P5","Prednisone","2024-01-03 08:00:00","2024-01-10 08:00:00","PO",40,"mg",1),
    # Follow-up at 105 days: phase-specific PJP context.
    (1001,2005,"P6","Tacrolimus","2024-04-15 08:00:00","2024-04-18 08:00:00","PO",2,"mg",2),
    (1001,2005,"P7","Trimethoprim-Sulfamethoxazole","2024-04-15 10:00:00","2024-04-18 10:00:00","PO",1,"tablet",1),
    # Prevalent/phase unknown: no phase-specific narrowing.
    (1002,2002,"P8","Tacrolimus","2024-02-01 08:00:00","2024-02-05 08:00:00","PO",2,"mg",2),
    (1002,2002,"P9","Fluconazole","2024-02-01 10:00:00","2024-02-04 10:00:00","PO",200,"mg",1),
    (1002,2002,"P10","Diphenhydramine","2024-02-01 11:00:00","2024-02-01 13:00:00","PO",25,"mg",1),
    (1002,2002,"P10B","Diazepam","2024-02-01 12:00:00","2024-02-02 12:00:00","PO",5,"mg",1),
    # Maintenance patient with time-scoped CRRT.
    (1003,2003,"P11","Tacrolimus","2024-03-01 08:00:00","2024-03-05 08:00:00","PO",2,"mg",2),
    (1003,2003,"P12","Fluconazole","2024-03-01 12:00:00","2024-03-04 12:00:00","PO",200,"mg",1),
    # Acute creatinine deterioration with renal screening signal.
    (1004,2004,"P13","Valganciclovir","2024-04-02 09:00:00","2024-04-05 09:00:00","PO",450,"mg",1),
    (1004,2004,"P14","Metformin","2024-04-02 10:00:00","2024-04-05 10:00:00","PO",500,"mg",2),
]
pd.DataFrame(
    prescriptions,
    columns=["subject_id","hadm_id","poe_id","drug","starttime","stoptime","route","dose_val_rx","dose_unit_rx","doses_per_24_hrs"],
).to_csv(HOSP / "prescriptions.csv", index=False)

labs = [
    (1001,2001,50912,"2024-01-01 07:30:00",1.1),
    (1001,2001,50912,"2024-01-02 08:30:00",1.2),
    (1001,2005,50912,"2024-04-15 07:30:00",1.2),
    (1002,2002,50912,"2024-02-01 08:00:00",1.0),
    (1003,2003,50912,"2024-03-01 08:00:00",2.0),
    (1003,2003,50912,"2024-03-01 11:00:00",2.2),
    (1004,2004,50912,"2024-04-01 08:00:00",1.0),
    (1004,2004,50912,"2024-04-02 08:00:00",1.4),
]
pd.DataFrame(labs, columns=["subject_id","hadm_id","itemid","charttime","valuenum"]).to_csv(HOSP / "labevents.csv", index=False)

pd.DataFrame(
    [
        (1001,2001,1,"0TY00Z0",10,"2024-01-01"),
        (1003,1999,1,"55.69",9,"2022-01-01"),
    ], columns=["subject_id","hadm_id","seq_num","icd_code","icd_version","chartdate"]
).to_csv(HOSP / "procedures_icd.csv", index=False)

pd.DataFrame(
    [
        ("0TY00Z0",10,"Transplantation of right kidney, allogeneic, open approach"),
        ("55.69",9,"Other kidney transplantation"),
    ], columns=["icd_code","icd_version","long_title"]
).to_csv(HOSP / "d_icd_procedures.csv", index=False)

pd.DataFrame(
    [(1001,2001,1,"Z94.0",10),(1002,2002,1,"Z94.0",10),(1003,2003,1,"Z94.0",10),(1004,2004,1,"Z94.0",10)],
    columns=["subject_id","hadm_id","seq_num","icd_code","icd_version"]
).to_csv(HOSP / "diagnoses_icd.csv", index=False)

pd.DataFrame(
    [
        (1001,2001,"2024-01-01 07:00:00","2024-01-10 12:00:00","TRANSPLANT"),
        (1001,2005,"2024-04-15 07:00:00","2024-04-18 12:00:00","WARD"),
        (1002,2002,"2024-02-01 07:00:00","2024-02-05 12:00:00","WARD"),
        (1003,2003,"2024-03-01 07:00:00","2024-03-05 12:00:00","ICU"),
        (1004,2004,"2024-04-01 07:00:00","2024-04-05 12:00:00","WARD"),
    ], columns=["subject_id","hadm_id","intime","outtime","careunit"]
).to_csv(HOSP / "transfers.csv", index=False)

pd.DataFrame(
    [
        (1001,2001,"2024-01-01 07:00:00","TRANSPLANT"),
        (1001,2005,"2024-04-15 07:00:00","MED"),
        (1002,2002,"2024-02-01 07:00:00","MED"),
        (1003,2003,"2024-03-01 07:00:00","MED"),
        (1004,2004,"2024-04-01 07:00:00","MED"),
    ], columns=["subject_id","hadm_id","transfertime","curr_service"]
).to_csv(HOSP / "services.csv", index=False)

pd.DataFrame(
    [(1003,2003,225802,"2024-03-01 10:00:00","2024-03-01 18:00:00",0,"FinishedRunning")],
    columns=["subject_id","hadm_id","itemid","starttime","endtime","cancelreason","statusdescription"]
).to_csv(ICU / "procedureevents.csv", index=False)

print(f"Synthetic fixtures written to {DEMO}")
