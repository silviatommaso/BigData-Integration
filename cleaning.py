import re

input_file = "tmd.csv"
output_file = "tmdb.csv"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# =========================
# STEP 1: swap sicuro ; <-> ,
# =========================

content = content.replace("§", "§§§")   # sicurezza

content = content.replace(";", "§")     # placeholder
content = content.replace(",", ";")
content = content.replace("§", ",")

# =========================
# STEP 2: correzione logica
# =========================

fixed_lines = []

for line in content.splitlines():
    parts = line.split(";")

    if len(parts) < 3:
        continue

    fixed = [parts[0], parts[1]]  # ID + title iniziale

    i = 2
    while i < len(parts):

        part = parts[i]

        # se è anno valido → resta campo separato
        if re.fullmatch(r"\d{4}", part):
            fixed.append(part)
        else:
            # NON è anno → deve tornare nel titolo
            fixed[1] += "," + part

        i += 1

    fixed_lines.append(";".join(fixed))

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(fixed_lines))

print("OK - completato")