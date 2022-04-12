# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Candidate section headers used to extract targeted sections in MIMIC III's discharge summary.

Varations are listed as List[Tuple[str, int]], where the str is the variation, and int is the frequency counts.
"""

past_medical_history_variations = [
    ('Past Medical History:', 38767),
    ('PAST MEDICAL HISTORY:', 9843),
    ('PMH:', 2814),
    ('PMHx:', 517),
    ('PMH', 75),
    ('Past Medical History', 47),
    ('PAST MEDICAL HISTORY', 31),
    ('PREVIOUS MEDICAL HISTORY:', 22),
    ('MEDICAL HISTORY:', 22),
    ('PMHX:', 20),
    ('PRIOR MEDICAL HISTORY:', 17),
    ('PAST MEDICAL AND SURGICAL HISTORY:', 15),
    ('BRIEF CLINICAL HISTORY:', 12),
    ('HISTORY AND PHYSICAL:', 11),
    ('PMHx', 8)
]

social_history_variations = [
    ('Social History:', 37784),
    ('SOCIAL HISTORY:', 6489),
    ('Social Hx:', 314),
    ('Social history:', 28),
    ('SOCIAL  HISTORY:', 14),
    ('Social hx:', 12)
]

family_history_variations = [
    ('Family History:', 37479),
    ('FAMILY HISTORY:', 3380),
    ('Family Hx:', 211),
    ('Family history:', 24),
    ('FAMILY PSYCHIATRIC HISTORY:', 15),
    ('FAMILY MEDICAL HISTORY:', 12),
    ('Family Medical History:', 10),
]

pertinent_results_variations = [
    ('Pertinent Results:', 37292),
    ('PERTINENT LABORATORY VALUES ON PRESENTATION:', 526),
    ('PERTINENT RADIOLOGY/IMAGING:', 363),
    ('PERTINENT LABORATORY DATA ON PRESENTATION:', 324),
    ('PERTINENT LABS:', 158),
    ('Pertinent Labs:', 73),
    ('Other pertinent labs:', 59),
    ('OTHER PERTINENT LABS:', 57),
    ('PERTINENT LABS/STUDIES:', 49),
    ('Pertinent labs:', 41),
    ('PERTINENT LABORATORY VALUES ON DISCHARGE:', 31),
    ('PERTINENT LABORATORIES:', 25),
    ('Pertinent Imaging:', 25),
    ('Pertinent Labs/Studies:', 25),
    ('PERTINENT LABORATORY DATA:', 23),
    ('Other Pertinent Labs:', 21),
    ('PERTINENT STUDIES:', 18),
    ('PERTINENT LABORATORY DATA ON ADMISSION:', 17),
    ('PERTINENT LABS DURING HOSPITALIZATION:', 17),
    ('PERTINENT LABS AND STUDIES:', 16),
    ('PERTINENT INTERVAL LABS:', 15),
    ('Other pertinent social history:', 15),
    ('PERTINENT IMAGING:', 14),
    ('PERTINENT LABORATORY DATA ON DISCHARGE:', 13),
    ('PERTINENT RESULTS:', 13),
    ('Pertinent studies:', 11),
    ('pertinent laboratory data:', 11)
]

history_of_present_illness_variations = [
    ('History of Present Illness:', 38145),
    ('HISTORY OF PRESENT ILLNESS:', 11105),
    ('HPI:', 1313),
    ('HPI', 0),
    ('HISTORY OF THE PRESENT ILLNESS:', 936),
    ('HISTORY OF PRESENTING ILLNESS:', 32),
    ('HISTORY OF PRESENTING ILLNESS', 29),
]

past_surgical_history_variations = [
    ('PAST SURGICAL HISTORY:', 2396),
    ('Past Surgical History:', 1121),
    ('PSHx:', 215),
    ('Past Surgical History', 205),
    ('SHx:', 53),
    ('PAST SURGICAL HISTORY', 43),
    ('Surgical History:', 34),
    ('SURGICAL HISTORY:', 32),
    ('Past Surgical Hx:', 10),
    ('PREVIOUS SURGICAL HISTORY:', 10),
    ('Surgical History', 9),
    ('Shx:', 9),
    ('PSHX:', 9),
    ('SHX:', 9),
    ('PRIOR SURGICAL HISTORY:', 8)
]

other_medical_history_variations = [
    ('Other Past Medical History:', 134),
    ('Other Past History:', 64),
    ('OTHER PAST MEDICAL HISTORY:', 61),
    ('OTHER MEDICAL HISTORY:', 40),
    ('Other PMH:', 26),
    ('Other Medical History:', 14),
    ('Other PMHx:', 13),
    ('OTHER PMH:', 12),
    ('OTHER PAST MEDICAL HISTORY', 8),
    ('OTHER PAST HISTORY:', 8)
]

cheif_complaint_variations = [
    ('Chief Complaint:', 38203),
    ('CHIEF COMPLAINT:', 2544),
    ('cc:', 591),
    ('CC:', 453),
    ('Chief complaint:', 11)
]

major_surgical_or_invasive_procedure_variations = [
    ('Major Surgical or Invasive Procedure:', 38024),
    ('MAJOR SURGICAL PROCEDURES:', 29),
    ('MAJOR SURGICAL OR INVASIVE PROCEDURES:', 26),
    ('MAJOR PROCEDURES:', 15),
    ('MAJOR SURGICAL/INVASIVE PROCEDURES:', 15),
    ('MAJOR SURGICAL INVASIVE PROCEDURES:', 14),
    ('MAJOR SURGICAL PROCEDURE:', 12),
    ('MAJOR SURGICAL AND INVASIVE PROCEDURES:', 11),
    ('MAJOR SURGICAL/INVASIVE PROCEDURES PERFORMED:', 10)
]

brief_hospital_course_variations = [
    ('Brief Hospital Course:', 38007),
    ('HOSPITAL COURSE:', 9450),
    ('SUMMARY OF HOSPITAL COURSE:', 377),
    ('BRIEF SUMMARY OF HOSPITAL COURSE:', 169),
    ('BRIEF HOSPITAL COURSE:', 152),
    ('Hospital Course:', 96),
]

hospital_course_by_systems_variations = [
    ('SUMMARY OF HOSPITAL COURSE BY SYSTEMS:', 641),
    ('HOSPITAL COURSE BY SYSTEMS:', 454),
    ('HOSPITAL COURSE BY SYSTEM:', 353),
    ('CONCISE SUMMARY OF HOSPITAL COURSE BY ISSUE/SYSTEM:', 167),
    ('SUMMARY OF HOSPITAL COURSE BY SYSTEM:', 149),
    ('HOSPITAL COURSE BY ISSUE/SYSTEM:', 145),
    ('HISTORY OF HOSPITAL COURSE BY SYSTEMS:', 143),
    ('CONCISE SUMMARY OF HOSPITAL COURSE:', 115)
]

medications_on_admission_variations = [
    ('Medications on Admission:', 37204),
    ('MEDICATIONS ON ADMISSION:', 3814),
    ('ADMISSION MEDICATIONS:', 454),
    ('MEDICATIONS PRIOR TO ADMISSION:', 170),
    ('Medications prior to admission:', 150),
    ('ADMITTING MEDICATIONS:', 63),
    ('Medications on admission:', 37),
    ('Meds on Admission:', 34),
    ('MEDICATIONS UPON ADMISSION:', 22),
    ('MEDS ON ADMISSION:', 20),
    ('Meds on admission:', 20),
    ('MEDICATIONS AT TIME OF ADMISSION:', 14),
    ('MEDICATIONS AT THE TIME OF ADMISSION:', 14),
    ('Admission Medications:', 10)
]

mapping_variations = {
    'past_medical_history': past_medical_history_variations,
    'history_of_present_illness': history_of_present_illness_variations,
    'chief_complaint': cheif_complaint_variations,
    'brief_hospital_course': brief_hospital_course_variations,
    'medications_on_admission': medications_on_admission_variations,
    'social_history': social_history_variations,
    'family_history': family_history_variations,
    # 'pertinent_results': pertinent_results_variations
    # 'past_surgical_history': past_surgical_history_variations,
    # 'other_medical_history': other_medical_history_variations,
    # 'major_surgical_or_invasive_procedure': major_surgical_or_invasive_procedure_variations,
    'hospital_course_by_systems': hospital_course_by_systems_variations
}
