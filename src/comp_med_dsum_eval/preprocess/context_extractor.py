# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Context Definition:

<Context> ::= <Hint><NoteType><Demographic><MedList><LabList><NoteList>
<Demographic> ::= <Gender> <Age>
<Hint> ::= first-line-of-note " <H> "
<NoteType> ::= note-type " <T> "
<Gender> ::= "M" | "F" " <G> "
<Age> ::= age-in-years " <A> "
<MedList> ::= List[<Medication>]
<Medication> ::= drug-name
<Delim> ::= " | "
<LabList> ::= List[<Lab>]
<Lab> ::= lab-name " , " lab-value " , " unit-of-measurement <LabFlag>
<LabFlag> ::= " abnormal " | ""
<NoteList> ::= List[<Note>]
<Note> ::= raw-note-text
"""


from pandas.errors import OutOfBoundsDatetime


class ContextExtractor:
    """
    ContextExtractor is a class used to extract context (prior notes and structure information) of a discharge summary
    """
    def __init__(self, target_section_type, patients, admissions, prior_notes, prescriptions, labs, d_labs,
                 demographic_flag, hint_flag, medList_flag, labList_flag, noteList_flag):
        """
        Initialized preprocessed data
        :param target_section_type:
        :param patients:
        :param admissions:
        :param prior_notes:
        :param prescriptions:
        :param labs:
        :param d_labs:
        :param demographic_flag:
        :param hint_flag:
        :param medList_flag:
        :param labList_flag:
        :param noteList_flag:
        """
        self.target_section_type = target_section_type
        self.patients = patients
        self.admissions = admissions
        self.prior_notes = prior_notes
        self.prescriptions = prescriptions
        self.labs = labs
        self.d_labs = d_labs
        self.demographic_flag = demographic_flag
        self.hint_flag = hint_flag
        self.medList_flag = medList_flag
        self.labList_flag = labList_flag
        self.noteList_flag = noteList_flag

    def get_context(self, target_section, target_time, encounter_id, subject_id):
        """
        Get context in dictionary form, see __doc__ for definition of
        <Hint><NoteType><Demographic><MedList><LabList><NoteList>
        :param target_section: str, the target medical section
        :param target_time: DateTime, discharge summary chart date
        :param encounter_id: int, id of the encounter (HADM_ID)
        :param subject_id: int, id of the patient (SUBJECT_ID)
        :return: Dict[str, Any], context in dictionary form
        """
        patients, admissions, prior_notes, prescriptions, labs = self.filter_by_id_and_time(subject_id,
                                                                                            encounter_id,
                                                                                            target_time)
        hint_str = self.get_Hint(target_section) if self.hint_flag else ''
        note_type_str = self.get_NoteType() if self.hint_flag else ''
        demographic_str = self.get_Demographic(patients, admissions) if self.demographic_flag else ''
        medList_list = self.get_med_list(prescriptions) if self.medList_flag else []
        labList_list = self.get_lab_list(labs) if self.labList_flag else []
        noteList_list = self.get_note_list(prior_notes) if self.noteList_flag else []
        return {'Hint': hint_str,
                'NoteType': note_type_str,
                'Demographic': demographic_str,
                'MedList': medList_list,
                'LabList': labList_list,
                'NoteList': noteList_list}

    def filter_by_id_and_time(self, subject_id, encounter_id, target_time):
        """
        Filter the raw dataframe by subject_id, encounter_id, and target_time
        :param subject_id: int, id of the patient (SUBJECT_ID)
        :param encounter_id: int, id of the encounter (HADM_ID)
        :param target_time: DateTime, discharge summary chart date
        :return: Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[Any]], pd.DataFrame, pd.DataFrame], filtered data
        """
        if self.demographic_flag:
            patients = self.patients.loc[self.patients.SUBJECT_ID == subject_id,
                                         ['SUBJECT_ID', 'GENDER', 'DOB']].iloc[0]
            admissions = self.admissions.loc[self.admissions.HADM_ID == encounter_id,
                                             ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']].iloc[0]
        else:
            patients = self.patients
            admissions = self.admissions

        if self.noteList_flag:
            if encounter_id in self.prior_notes:
                prior_notes = self.prior_notes[encounter_id]
            else:
                prior_notes = {'TEXT': [], 'CATEGORY': [], 'process_text': []}
        else:
            prior_notes = self.prior_notes

        if self.medList_flag:
            prescriptions = self.prescriptions.loc[self.prescriptions.HADM_ID == encounter_id]
            prescriptions = prescriptions.loc[prescriptions.STARTDATE < target_time,
                                              ['STARTDATE', 'DRUG']]
        else:
            prescriptions = self.prescriptions

        if self.labList_flag:
            labs = self.labs.loc[self.labs.HADM_ID == encounter_id]
            labs = labs.loc[labs.CHARTTIME < target_time,
                            ['CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM', 'FLAG']].join(
                self.d_labs[['LABEL']], on='ITEMID')
        else:
            labs = self.labs

        return patients, admissions, prior_notes, prescriptions, labs

    def get_Hint(self, target_section):
        return self.get_first_line_of_note(target_section) + " <H> "

    def get_first_line_of_note(self, note_text_list):
        return note_text_list[0]

    def get_NoteType(self):
        return self.target_section_type + " <T> "

    def get_Demographic(self, patients, admissions):
        return self.get_Gender(patients) + self.get_Age(patients, admissions)

    def get_Gender(self, patients):
        return self.get_gender(patients) + " <G> "

    def get_gender(self, patients):
        return patients.GENDER

    def get_Age(self, patients, admissions):
        return self.get_age_in_years(patients, admissions) + " <A> "

    def get_age_in_years(self, patients, admissions):
        try:
            return str(int((admissions.ADMITTIME - patients.DOB).days / 365.2425))
        except OutOfBoundsDatetime:
            return "invalid"

    def get_med_list(self, prescriptions):
        return prescriptions.sort_values('STARTDATE')['DRUG'].fillna('').to_list()

    def get_lab_list(self, labs):
        return labs.sort_values('CHARTTIME')[
            ['LABEL', 'VALUE', 'VALUEUOM', 'FLAG']].fillna('').apply(
            lambda x: ' , '.join(map(str, x)), axis=1).to_list()

    def get_note_list(self, prior_notes):
        return prior_notes['process_text']
