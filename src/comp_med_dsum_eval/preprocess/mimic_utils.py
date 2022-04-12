# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re


def _pattern_repl(matchobj):
    """
    :param matchobj: re.Match object
    :return: Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))


def create_note_tok(note_str, prefix=True):
    note_str_clean = re.sub('[\W+]+', '_', note_str.strip()).lower()
    if prefix:
        return 'note_type_' + note_str_clean
    return note_str_clean


def add_note_id(notes_df):
    note_id_cols = ['row_id', 'visit_id', 'patient_id']
    print('Compiling note_id column...')
    notes_df['note_id'] = notes_df[note_id_cols].apply(
        lambda row: 'mimic-' + '-'.join([str(int(x)) for x in row.values.astype(str)]), axis=1)


def patient_id_from_note_id(note_id):
    return note_id.split('-')[-1]


def clean_mimic(text):
    """
    :param text: string representing raw MIMIC note
    :return: cleaned string
    - Replace [**Patterns**] with spaces
    """
    cleaned = []
    for line in text.split('\n'):
        line_strip = line.strip()
        line_strip = re.sub(r'\[\*\*.*?\*\*]', _pattern_repl, line_strip)
        # Replace `_` with spaces.
        line_strip = re.sub(r'[_*?/()]+', ' ', line_strip)
        line_strip = re.sub(r'[<>]+', ' ', line_strip)  # this is a special character for us (don't confuse)
        line_strip = re.sub(r'\s+', ' ', line_strip).strip()
        if len(line_strip) > 0:
            cleaned.append(line_strip)
    cleaned_str = '\n'.join(cleaned)
    # BRIEF HOSPITAL COURSE     : patient ... -> BRIEF HOSPITAL COURSE: patient ...
    cleaned_str_no_space_before_colon = re.sub(r'\s+:', ':', cleaned_str)
    return cleaned_str_no_space_before_colon


def remove_identifiers(text):
    '''
    Removes MIMIC markers of deidentified info from the text.
    Replaces with single, all-caps words such as ID, AGE, PHONE, etc.
    '''
    regex = r"\[\*\*.{0,15}%s.*?\*\*\]"

    text = re.sub(regex % "serial number", 'phi_sn', text, flags=re.I)
    text = re.sub(regex % "identifier", 'phi_id', text, flags=re.I)
    text = re.sub(regex % "medical record number", 'phi_mrn', text, flags=re.I)
    text = re.sub(regex % "social security number", 'phi_ssn', text, flags=re.I)

    # AGE
    text = re.sub(regex % "age", 'phi_age', text, flags=re.I)

    # PHONE
    text = re.sub(regex % "phone", 'phi_phone', text, flags=re.I)
    text = re.sub(regex % "pager number", 'phi_pager', text, flags=re.I)
    text = re.sub(regex % "contact info", 'phi_contact', text, flags=re.I)
    text = re.sub(regex % "provider number", 'phi_provider_phone', text, flags=re.I)

    # NAME
    text = re.sub(regex % "name", 'phi_name', text, flags=re.I)
    text = re.sub(regex % "dictator", 'phi_dictator', text, flags=re.I)
    text = re.sub(regex % "attending", 'phi_attending', text, flags=re.I)

    # HOSPITAL
    text = re.sub(regex % "hospital", 'phi_hospital', text, flags=re.I)

    # LOC
    text = re.sub(regex % "location", 'phi_loc', text, flags=re.I)
    text = re.sub(regex % "address", 'phi_address', text, flags=re.I)
    text = re.sub(regex % "country", 'phi_country', text, flags=re.I)
    text = re.sub(regex % "state", 'phi_state', text, flags=re.I)
    text = re.sub(regex % "university", 'phi_uni', text, flags=re.I)

    # DATE
    text = re.sub(regex % "year", 'phi_year', text, flags=re.I)
    text = re.sub(regex % "month", 'phi_month', text, flags=re.I)
    text = re.sub(regex % "date", 'phi_dt', text, flags=re.I)
    text = re.sub(r"\[?\*\*[0-9]{0,2}/[0-9]{0,4}\*\*\]?", 'phi_month_year', text, flags=re.I)  # e.g. 03/1990
    text = re.sub(r"\[?\*\*[0-9]{0,4}\*\*\]?", 'phi_year', text, flags=re.I)  # e.g. 1991
    text = re.sub(r"\[?\*\*(?:[0-9]{0,4}-)?[0-9]{0,2}-[0-9]{0,2}\*\*\]?", 'phi_month_day_year', text, flags=re.I)

    # CLIP
    text = re.sub(r"\[?\*\*.*Clip Number.*\*\*\]?", 'phi_clip_num', text, flags=re.I)

    # HOLIDAY
    text = re.sub(r"\[?\*\*.*Holiday.*\*\*\]?", 'phi_holiday', text, flags=re.I)

    # COMPANY
    text = re.sub(r"\[?\*\*.*Company.*\*\*\]?", 'phi_company', text, flags=re.I)

    # JOB
    text = re.sub(r"\[?\*\*.*Job Number.*\*\*\]?", 'phi_job_num', text, flags=re.I)

    # UNIT_NUM
    text = re.sub(r"\[?\*\*.*Unit Number.*\*\*\]?", 'phi_unit_num', text, flags=re.I)

    # URL
    text = re.sub(r"\[?\*\*.*url.*\*\*\]?", 'phi_url', text, flags=re.I)

    # OTHER
    text = re.sub(r"\[?\*\*.*\d+.*\*\*\]?", 'phi_other', text, flags=re.I)
    text = re.sub(r"\[?\*\* +\*\*\]?", 'phi_other', text, flags=re.I)

    return text


def add_note_id(notes_df):
    note_id_cols = ['ROW_ID', 'HADM_ID', 'SUBJECT_ID']
    print('Compiling note_id column...')
    notes_df['note_id'] = notes_df[note_id_cols].apply(
        lambda row: 'mimic-' + '-'.join([str(x).split('.')[0] for x in row.values.astype(str)]), axis=1)
