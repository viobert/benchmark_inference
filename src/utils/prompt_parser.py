import re


def parse_prompt(txt: str) -> dict:
    system = re.search(r"<SYSTEM>(.*?)</SYSTEM>", txt, re.S)
    user = re.search(r"<USER>(.*?)</USER>", txt, re.S)

    return {
        "system": system.group(1).strip() if system else "",
        "user": user.group(1).strip() if user else txt.strip(),
    }
