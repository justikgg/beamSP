import re

_NAME_MAP = {
    "a_h0": r"(a/h_0)",
    "is_steel": r"M",
    "H": r"H",
    "s": r"s",
    "R": r"R",
    "E": r"E",
}


def _replace_func_brackets(s, func, open_repl, close_repl="}"):
    out = []
    i = 0
    token = func + "("
    while i < len(s):
        if s.startswith(token, i):
            out.append(open_repl)
            i += len(token)
            depth = 1
            while i < len(s) and depth > 0:
                if s[i] == "(":
                    depth += 1
                    out.append("(")
                elif s[i] == ")":
                    depth -= 1
                    out.append(close_repl if depth == 0 else ")")
                else:
                    out.append(s[i])
                i += 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def to_mathtext(expr):
    s = expr
    s = re.sub(r"(\d\.?\d*)[eE]([+-]?\d+)", lambda m: f"{m.group(1)}\\times10^{{{int(m.group(2))}}}", s)
    for name in sorted(_NAME_MAP, key=len, reverse=True):
        s = re.sub(r"(?<![\w]){}(?![\w])".format(re.escape(name)), _NAME_MAP[name].replace("\\", "\\\\"), s)
    guard = 0
    while "sqrt(" in s and guard < 50:
        s = _replace_func_brackets(s, "sqrt", r"\sqrt{", "}")
        guard += 1
    s = s.replace("ln(", r"\ln(")

    s = re.sub(r"\^(-?\d+\.?\d*)", lambda m: f"^{{{m.group(1)}}}", s)

    s = s.replace("*", r"\cdot ")

    s = re.sub(r"Qдв(?:_(\w+))?",
               lambda m: "Q_{дв" + ("," + m.group(1).replace("_", r"\_") if m.group(1) else "") + "}", s)

    return f"${s}$"
