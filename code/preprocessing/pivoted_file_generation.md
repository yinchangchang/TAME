# SQL for pivoted\_\*.csv generation

```
-- Drop table

-- DROP TABLE public.pivoted_sofa;

CREATE TABLE public.pivoted_sofa (
	icustay_id int4 NULL,
	hr int4 NULL,
	starttime timestamp NULL,
	endtime timestamp NULL,
	pao2fio2ratio_novent float8 NULL,
	pao2fio2ratio_vent float8 NULL,
	rate_epinephrine float8 NULL,
	rate_norepinephrine float8 NULL,
	rate_dopamine float8 NULL,
	rate_dobutamine float8 NULL,
	meanbp_min float8 NULL,
	gcs_min float8 NULL,
	urineoutput float8 NULL,
	bilirubin_max float8 NULL,
	creatinine_max float8 NULL,
	platelet_min float8 NULL,
	respiration int2 NULL,
	coagulation int2 NULL,
	liver int2 NULL,
	cardiovascular int2 NULL,
	cns int2 NULL,
	renal int2 NULL,
	respiration_24hours int2 NULL,
	coagulation_24hours int2 NULL,
	liver_24hours int2 NULL,
	cardiovascular_24hours int2 NULL,
	cns_24hours int2 NULL,
	renal_24hours int2 NULL,
	sofa_24hours int2 NULL
);






CREATE MATERIALIZED VIEW public.pivoted_lab
TABLESPACE pg_default
AS WITH i AS (
         SELECT icustays.subject_id,
            icustays.icustay_id,
            icustays.intime,
            icustays.outtime,
            lag(icustays.outtime) OVER (PARTITION BY icustays.subject_id ORDER BY icustays.intime) AS outtime_lag,
            lead(icustays.intime) OVER (PARTITION BY icustays.subject_id ORDER BY icustays.intime) AS intime_lead
           FROM icustays
        ), iid_assign AS (
         SELECT i.subject_id,
            i.icustay_id,
                CASE
                    WHEN i.outtime_lag IS NOT NULL AND i.outtime_lag > (i.intime - '24:00:00'::interval hour) THEN i.intime - (i.intime - i.outtime_lag) / 2::double precision
                    ELSE i.intime - '12:00:00'::interval hour
                END AS data_start,
                CASE
                    WHEN i.intime_lead IS NOT NULL AND i.intime_lead < (i.outtime + '24:00:00'::interval hour) THEN i.outtime + (i.intime_lead - i.outtime) / 2::double precision
                    ELSE i.outtime + '12:00:00'::interval hour
                END AS data_end
           FROM i
        ), h AS (
         SELECT admissions.subject_id,
            admissions.hadm_id,
            admissions.admittime,
            admissions.dischtime,
            lag(admissions.dischtime) OVER (PARTITION BY admissions.subject_id ORDER BY admissions.admittime) AS dischtime_lag,
            lead(admissions.admittime) OVER (PARTITION BY admissions.subject_id ORDER BY admissions.admittime) AS admittime_lead
           FROM admissions
        ), adm AS (
         SELECT h.subject_id,
            h.hadm_id,
                CASE
                    WHEN h.dischtime_lag IS NOT NULL AND h.dischtime_lag > (h.admittime - '24:00:00'::interval hour) THEN h.admittime - (h.admittime - h.dischtime_lag) / 2::double precision
                    ELSE h.admittime - '12:00:00'::interval hour
                END AS data_start,
                CASE
                    WHEN h.admittime_lead IS NOT NULL AND h.admittime_lead < (h.dischtime + '24:00:00'::interval hour) THEN h.dischtime + (h.admittime_lead - h.dischtime) / 2::double precision
                    ELSE h.dischtime + '12:00:00'::interval hour
                END AS data_end
           FROM h
        ), le AS (
         SELECT labevents.subject_id,
            labevents.charttime,
                CASE
                    WHEN labevents.itemid = 50868 THEN 'ANION GAP'::text
                    WHEN labevents.itemid = 50862 THEN 'ALBUMIN'::text
                    WHEN labevents.itemid = 51144 THEN 'BANDS'::text
                    WHEN labevents.itemid = 50882 THEN 'BICARBONATE'::text
                    WHEN labevents.itemid = 50885 THEN 'BILIRUBIN'::text
                    WHEN labevents.itemid = 50912 THEN 'CREATININE'::text
                    WHEN labevents.itemid = 50902 THEN 'CHLORIDE'::text
                    WHEN labevents.itemid = 50931 THEN 'GLUCOSE'::text
                    WHEN labevents.itemid = 51221 THEN 'HEMATOCRIT'::text
                    WHEN labevents.itemid = 51222 THEN 'HEMOGLOBIN'::text
                    WHEN labevents.itemid = 50813 THEN 'LACTATE'::text
                    WHEN labevents.itemid = 51265 THEN 'PLATELET'::text
                    WHEN labevents.itemid = 50971 THEN 'POTASSIUM'::text
                    WHEN labevents.itemid = 51275 THEN 'PTT'::text
                    WHEN labevents.itemid = 51237 THEN 'INR'::text
                    WHEN labevents.itemid = 51274 THEN 'PT'::text
                    WHEN labevents.itemid = 50983 THEN 'SODIUM'::text
                    WHEN labevents.itemid = 51006 THEN 'BUN'::text
                    WHEN labevents.itemid = 51300 THEN 'WBC'::text
                    WHEN labevents.itemid = 51301 THEN 'WBC'::text
                    ELSE NULL::text
                END AS label,
                CASE
                    WHEN labevents.itemid = 50862 AND labevents.valuenum > 10::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 50868 AND labevents.valuenum > 10000::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 51144 AND labevents.valuenum < 0::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 51144 AND labevents.valuenum > 100::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 50882 AND labevents.valuenum > 10000::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 50885 AND labevents.valuenum > 150::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 50806 AND labevents.valuenum > 10000::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 50902 AND labevents.valuenum > 10000::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 50912 AND labevents.valuenum > 150::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 50809 AND labevents.valuenum > 10000::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 50931 AND labevents.valuenum > 10000::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 50810 AND labevents.valuenum > 100::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 51221 AND labevents.valuenum > 100::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 50811 AND labevents.valuenum > 50::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 51222 AND labevents.valuenum > 50::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 50813 AND labevents.valuenum > 50::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 51265 AND labevents.valuenum > 10000::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 50822 AND labevents.valuenum > 30::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 50971 AND labevents.valuenum > 30::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 51275 AND labevents.valuenum > 150::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 51237 AND labevents.valuenum > 50::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 51274 AND labevents.valuenum > 150::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 50824 AND labevents.valuenum > 200::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 50983 AND labevents.valuenum > 200::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 51006 AND labevents.valuenum > 300::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 51300 AND labevents.valuenum > 1000::double precision THEN NULL::double precision
                    WHEN labevents.itemid = 51301 AND labevents.valuenum > 1000::double precision THEN NULL::double precision
                    ELSE labevents.valuenum
                END AS valuenum
           FROM labevents
          WHERE (labevents.itemid = ANY (ARRAY[50868, 50862, 51144, 50882, 50885, 50912, 50902, 50931, 51221, 51222, 50813, 51265, 50971, 51275, 51237, 51274, 50983, 51006, 51301, 51300])) AND labevents.valuenum IS NOT NULL AND labevents.valuenum > 0::double precision
        ), le_avg AS (
         SELECT le.subject_id,
            le.charttime,
            avg(
                CASE
                    WHEN le.label = 'ANION GAP'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS aniongap,
            avg(
                CASE
                    WHEN le.label = 'ALBUMIN'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS albumin,
            avg(
                CASE
                    WHEN le.label = 'BANDS'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS bands,
            avg(
                CASE
                    WHEN le.label = 'BICARBONATE'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS bicarbonate,
            avg(
                CASE
                    WHEN le.label = 'BILIRUBIN'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS bilirubin,
            avg(
                CASE
                    WHEN le.label = 'CREATININE'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS creatinine,
            avg(
                CASE
                    WHEN le.label = 'CHLORIDE'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS chloride,
            avg(
                CASE
                    WHEN le.label = 'GLUCOSE'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS glucose,
            avg(
                CASE
                    WHEN le.label = 'HEMATOCRIT'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS hematocrit,
            avg(
                CASE
                    WHEN le.label = 'HEMOGLOBIN'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS hemoglobin,
            avg(
                CASE
                    WHEN le.label = 'LACTATE'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS lactate,
            avg(
                CASE
                    WHEN le.label = 'PLATELET'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS platelet,
            avg(
                CASE
                    WHEN le.label = 'POTASSIUM'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS potassium,
            avg(
                CASE
                    WHEN le.label = 'PTT'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS ptt,
            avg(
                CASE
                    WHEN le.label = 'INR'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS inr,
            avg(
                CASE
                    WHEN le.label = 'PT'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS pt,
            avg(
                CASE
                    WHEN le.label = 'SODIUM'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS sodium,
            avg(
                CASE
                    WHEN le.label = 'BUN'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS bun,
            avg(
                CASE
                    WHEN le.label = 'WBC'::text THEN le.valuenum
                    ELSE NULL::double precision
                END) AS wbc
           FROM le
          GROUP BY le.subject_id, le.charttime
        )
 SELECT iid.icustay_id,
    adm.hadm_id,
    le_avg.subject_id,
    le_avg.charttime,
    le_avg.aniongap,
    le_avg.albumin,
    le_avg.bands,
    le_avg.bicarbonate,
    le_avg.bilirubin,
    le_avg.creatinine,
    le_avg.chloride,
    le_avg.glucose,
    le_avg.hematocrit,
    le_avg.hemoglobin,
    le_avg.lactate,
    le_avg.platelet,
    le_avg.potassium,
    le_avg.ptt,
    le_avg.inr,
    le_avg.pt,
    le_avg.sodium,
    le_avg.bun,
    le_avg.wbc
   FROM le_avg
     LEFT JOIN adm ON le_avg.subject_id = adm.subject_id AND le_avg.charttime >= adm.data_start AND le_avg.charttime < adm.data_end
     LEFT JOIN iid_assign iid ON le_avg.subject_id = iid.subject_id AND le_avg.charttime >= iid.data_start AND le_avg.charttime < iid.data_end
  ORDER BY le_avg.subject_id, le_avg.charttime
WITH DATA;


CREATE MATERIALIZED VIEW public.pivoted_vital
TABLESPACE pg_default
AS WITH ce AS (
         SELECT ce_1.icustay_id,
            ce_1.charttime,
                CASE
                    WHEN (ce_1.itemid = ANY (ARRAY[211, 220045])) AND ce_1.valuenum > 0::double precision AND ce_1.valuenum < 300::double precision THEN ce_1.valuenum
                    ELSE NULL::double precision
                END AS heartrate,
                CASE
                    WHEN (ce_1.itemid = ANY (ARRAY[51, 442, 455, 6701, 220179, 220050])) AND ce_1.valuenum > 0::double precision AND ce_1.valuenum < 400::double precision THEN ce_1.valuenum
                    ELSE NULL::double precision
                END AS sysbp,
                CASE
                    WHEN (ce_1.itemid = ANY (ARRAY[8368, 8440, 8441, 8555, 220180, 220051])) AND ce_1.valuenum > 0::double precision AND ce_1.valuenum < 300::double precision THEN ce_1.valuenum
                    ELSE NULL::double precision
                END AS diasbp,
                CASE
                    WHEN (ce_1.itemid = ANY (ARRAY[456, 52, 6702, 443, 220052, 220181, 225312])) AND ce_1.valuenum > 0::double precision AND ce_1.valuenum < 300::double precision THEN ce_1.valuenum
                    ELSE NULL::double precision
                END AS meanbp,
                CASE
                    WHEN (ce_1.itemid = ANY (ARRAY[615, 618, 220210, 224690])) AND ce_1.valuenum > 0::double precision AND ce_1.valuenum < 70::double precision THEN ce_1.valuenum
                    ELSE NULL::double precision
                END AS resprate,
                CASE
                    WHEN (ce_1.itemid = ANY (ARRAY[223761, 678])) AND ce_1.valuenum > 70::double precision AND ce_1.valuenum < 120::double precision THEN (ce_1.valuenum - 32::double precision) / 1.8::double precision
                    WHEN (ce_1.itemid = ANY (ARRAY[223762, 676])) AND ce_1.valuenum > 10::double precision AND ce_1.valuenum < 50::double precision THEN ce_1.valuenum
                    ELSE NULL::double precision
                END AS tempc,
                CASE
                    WHEN (ce_1.itemid = ANY (ARRAY[646, 220277])) AND ce_1.valuenum > 0::double precision AND ce_1.valuenum <= 100::double precision THEN ce_1.valuenum
                    ELSE NULL::double precision
                END AS spo2,
                CASE
                    WHEN (ce_1.itemid = ANY (ARRAY[807, 811, 1529, 3745, 3744, 225664, 220621, 226537])) AND ce_1.valuenum > 0::double precision THEN ce_1.valuenum
                    ELSE NULL::double precision
                END AS glucose
           FROM chartevents ce_1
          WHERE ce_1.error IS DISTINCT FROM 1 AND (ce_1.itemid = ANY (ARRAY[211, 220045, 51, 442, 455, 6701, 220179, 220050, 8368, 8440, 8441, 8555, 220180, 220051, 456, 52, 6702, 443, 220052, 220181, 225312, 618, 615, 220210, 224690, 646, 220277, 807, 811, 1529, 3745, 3744, 225664, 220621, 226537, 223762, 676, 223761, 678]))
        )
 SELECT ce.icustay_id,
    ce.charttime,
    avg(ce.heartrate) AS heartrate,
    avg(ce.sysbp) AS sysbp,
    avg(ce.diasbp) AS diasbp,
    avg(ce.meanbp) AS meanbp,
    avg(ce.resprate) AS resprate,
    avg(ce.tempc) AS tempc,
    avg(ce.spo2) AS spo2,
    avg(ce.glucose) AS glucose
   FROM ce
  GROUP BY ce.icustay_id, ce.charttime
  ORDER BY ce.icustay_id, ce.charttime
WITH DATA;



```
