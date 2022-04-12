# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

"""
HEADER STUFF
"""
# Building Blocks
MAX_HEADER_LEN = 50
BEGINNING = r'(?:^|\s{4,}|\t|\n)'
ALL_CAPS = r'[A-Z][A-Z0-9/() ]{0,' + str(MAX_HEADER_LEN - 2) + '}[A-Z)]'
MIXED_CAPS = r'[A-Z][A-z0-9/() ]{0,' + str(MAX_HEADER_LEN - 2) + '}[A-z)]'
NEWLINE = r'\s*\n'
COLON = r':+'

# 3 Different header styles
MIXED_CAPS_COURSE = r'(?:Hospital|ER|CLINIC|OR|ED|ICU|MICU|SICU|Post[- ]?[Oo]p)? ?Course(?!,)'
LOWER_CASE_COURSE = r'(?:hospital|er|clinic|or|ed|icu|micu|sicu|post[- ]?op)? ?course:+'
MIXED_CAPS_COLON = r'{}{}'.format(MIXED_CAPS, COLON)
BEGIN_MIXED_CAPS_COLON = r'{}{}{}'.format(BEGINNING, MIXED_CAPS, COLON)
BEGIN_SHORT_MIXED_CAPS_NEWLINE = r'{}{}{}'.format(BEGINNING, r'[A-Z][A-z]+\s?[A-z]+', NEWLINE)
ALL_CAPS_NEWLINE = r'{}{}'.format(ALL_CAPS, NEWLINE)
BEGIN_ALL_CAPS_NEWLINE = r'{}{}{}'.format(BEGINNING, ALL_CAPS, NEWLINE)
ALL_CAPS_COLON = r'{}{}'.format(ALL_CAPS, COLON)

HTML_REGEX = r' ?(<[a-z][^>]+>|<\/?[a-z]>) ?'
HTML_REGEX_NO_SPACE = r'(<[a-z][^>]+>|<\/?[a-z]>)'
HEADER_REGEX = r'({}|{}|{}|{}|{})'.format(
    BEGIN_MIXED_CAPS_COLON, BEGIN_ALL_CAPS_NEWLINE, ALL_CAPS_COLON, MIXED_CAPS_COURSE, LOWER_CASE_COURSE)
ONLY_INLINE_HEADER_REGEX = r'({}|{}|{}|{})'.format(
    MIXED_CAPS_COLON, ALL_CAPS_NEWLINE, ALL_CAPS_COLON, BEGIN_SHORT_MIXED_CAPS_NEWLINE)
SEP_REGEX = r'\.\s|\n{2,}|^\s{0,}\d{1,2}\s{0,}[-).]\s{1,}'

NEWLINE_REGEX = r'\n+[-.#]+'
LIST_REGEX = r'\s+\d\)|\d\)\s+|\s+\d\.\s+'
LONG_DELIMS = r'\-+ | \-+'
SUB_HEADERS = r' (?=[A-z]+:)'
EOS_PUNC = {'.', '!', '?'}

ACM_ENT_TYPES = ['DX_NAME', 'TEST_NAME', 'GENERIC_NAME', 'PROCEDURE_NAME', 'TREATMENT_NAME', 'BRAND_NAME']
