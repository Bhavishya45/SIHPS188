"""
ICAO Doc 9303 Machine Readable Zone (MRZ) parser + checksum validator.
TD3 format (passports): two lines of 44 characters.

This is a REAL implementation of the ICAO 9303 check-digit algorithm,
not a stub. It is used both by the Python backend (app.py) and ported
line-for-line into the dashboard's browser demo (see dashboard.html)
so the two stay logically identical.
"""

from dataclasses import dataclass, field
from typing import List, Optional

_WEIGHTS = [7, 3, 1]


def _char_value(c: str) -> int:
    """A=10 ... Z=35, 0-9=digit, '<'=0, per ICAO 9303 Appendix A."""
    if c == "<":
        return 0
    if c.isdigit():
        return int(c)
    if c.isalpha():
        return ord(c.upper()) - ord("A") + 10
    raise ValueError(f"Invalid MRZ character: {c!r}")


def check_digit(data: str) -> int:
    total = 0
    for i, c in enumerate(data):
        total += _char_value(c) * _WEIGHTS[i % 3]
    return total % 10


@dataclass
class MRZField:
    name: str
    raw: str
    expected_check: Optional[str] = None
    computed_check: Optional[int] = None
    valid: Optional[bool] = None


@dataclass
class MRZResult:
    document_type: str = ""
    issuing_country: str = ""
    surname: str = ""
    given_names: str = ""
    passport_number: str = ""
    nationality: str = ""
    date_of_birth: str = ""
    sex: str = ""
    expiry_date: str = ""
    personal_number: str = ""
    fields: List[MRZField] = field(default_factory=list)
    composite_valid: bool = False
    overall_valid: bool = False
    errors: List[str] = field(default_factory=list)


def _clean_line(line: str) -> str:
    line = line.strip().upper().replace(" ", "")
    return line.ljust(44, "<")[:44]


def parse_td3(line1: str, line2: str) -> MRZResult:
    """Parse + validate a TD3 (passport) MRZ. Returns MRZResult with
    every checksum verified individually plus the composite check."""
    result = MRZResult()

    if len(line1.strip()) < 5 or len(line2.strip()) < 5:
        result.errors.append("MRZ lines too short / not detected")
        return result

    l1 = _clean_line(line1)
    l2 = _clean_line(line2)

    # ---- Line 1: P<COUNTRY SURNAME<<GIVEN<NAMES<<<<... ----
    result.document_type = l1[0:2].replace("<", "")
    result.issuing_country = l1[2:5].replace("<", "")
    names_field = l1[5:44]
    if "<<" in names_field:
        surname, given = names_field.split("<<", 1)
    else:
        surname, given = names_field, ""
    result.surname = surname.replace("<", " ").strip()
    result.given_names = given.replace("<", " ").strip()

    # ---- Line 2: PASSPORTNO+CHK NATIONALITY DOB+CHK SEX EXP+CHK PERSONALNO+CHK COMPOSITE+CHK ----
    passport_no = l2[0:9]
    passport_chk = l2[9]
    nationality = l2[10:13]
    dob = l2[13:19]
    dob_chk = l2[19]
    sex = l2[20]
    expiry = l2[21:27]
    expiry_chk = l2[27]
    personal_no = l2[28:42]
    personal_chk = l2[42]
    composite_chk = l2[43]

    result.passport_number = passport_no.replace("<", "")
    result.nationality = nationality.replace("<", "")
    result.date_of_birth = dob
    result.sex = sex
    result.expiry_date = expiry
    result.personal_number = personal_no.replace("<", "")

    def add_field(name, raw, expected):
        try:
            computed = check_digit(raw)
            valid = str(computed) == expected
        except ValueError:
            computed = None
            valid = False
        result.fields.append(MRZField(name, raw, expected, computed, valid))
        return valid

    ok_passport = add_field("Passport number", passport_no, passport_chk)
    ok_dob = add_field("Date of birth", dob, dob_chk)
    ok_expiry = add_field("Expiry date", expiry, expiry_chk)
    ok_personal = add_field(
        "Personal number", personal_no, personal_chk
    ) if personal_no.replace("<", "") else True

    composite_data = passport_no + passport_chk + dob + dob_chk + expiry + expiry_chk + personal_no + personal_chk
    computed_composite = check_digit(composite_data)
    result.composite_valid = str(computed_composite) == composite_chk
    result.fields.append(
        MRZField("Composite", composite_data, composite_chk, computed_composite, result.composite_valid)
    )

    result.overall_valid = all([ok_passport, ok_dob, ok_expiry, ok_personal, result.composite_valid])

    if not ok_passport:
        result.errors.append("Passport number checksum mismatch")
    if not ok_dob:
        result.errors.append("Date-of-birth checksum mismatch")
    if not ok_expiry:
        result.errors.append("Expiry-date checksum mismatch")
    if not ok_personal:
        result.errors.append("Personal-number checksum mismatch")
    if not result.composite_valid:
        result.errors.append("Composite checksum mismatch (line-2 tampering indicator)")

    return result


if __name__ == "__main__":
    # Known-good ICAO sample MRZ (from Doc 9303 worked examples)
    l1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<"
    l2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
    r = parse_td3(l1, l2)
    print("Overall valid:", r.overall_valid)
    for f in r.fields:
        print(f" {f.name:18s} raw={f.raw:16s} expected={f.expected_check} computed={f.computed_check} valid={f.valid}")
