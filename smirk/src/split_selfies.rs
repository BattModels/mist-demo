use crate::split_smiles::{ATOMIC_SYMBOLS, BONDS, BRACKETED, CHARGE_OR_COUNT};
use const_format::concatcp;
use once_cell::sync::Lazy;
use regex::Regex;

// Capture Rings and Branching and Stereochemistry
const STRUCTURE: &'static str = r"Ring|Branch|@{1,2}";

// Capture Hydrogen
const HCOUNT: &'static str = r"[H]\d+";

// Capture tokens outside of brackets
pub static MATCH_OUTER: Lazy<Regex> =
    Lazy::new(|| Regex::new(concatcp!(BRACKETED, "|", r"\.")).unwrap());

// Capture tokens within brackets
pub static MATCH_INNER: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concatcp!(
        STRUCTURE,
        "|",
        BONDS,
        "|",
        ATOMIC_SYMBOLS,
        "|",
        CHARGE_OR_COUNT,
        "|",
        HCOUNT
    ))
    .unwrap()
});

pub fn split_chemically_consistent(a: &str) -> Vec<String> {
    // Iterate over string, splitting into tokens
    let mut tokens = Vec::new();
    for m in MATCH_OUTER.find_iter(a).map(|m| m.as_str()) {
        if m.starts_with("[") {
            tokens.push("[".to_string());
            for i in MATCH_INNER.find_iter(m) {
                tokens.push(i.as_str().to_string())
            }
            tokens.push("]".to_string());
        } else {
            tokens.push(m.to_string())
        }
    }
    return tokens;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_split(smile: &str) -> Vec<String> {
        let split = split_chemically_consistent(&smile);
        assert_eq!(split.join(""), smile);
        return split;
    }

    #[test]
    fn test_outer() {
        let split = check_split("[C][N][=C][=O]");
        assert_eq!(
            split,
            ["[", "C", "]", "[", "N", "]", "[", "=", "C", "]", "[", "=", "O", "]"]
        );
    }

    #[test]
    fn pyrrole() {
        let split = check_split("[NH1][C][=C][C][=C][Ring1][Branch1]");
        assert_eq!(
            split,
            [
                "[", "N", "H", "1", "]", "[", "C", "]", "[", "=", "C", "]", "[", "C", "]", "[",
                "=", "C", "]", "[", "Ring", "1", "]", "[", "Branch", "1", "]"
            ]
        );
    }

    #[test]
    fn test_brackets() {
        let split = check_split("[14C]");
        assert_eq!(split, ["[", "1", "4", "C", "]"]);
    }

    #[test]
    fn test_wikipedia_examples() {
        check_split("[N][#N]");
        check_split("[C][N][=C][=O]");
        check_split("[Cu+2].[O-1][S][=Branch1][C][=O][=Branch1][C][=O][O-1]");
        check_split("[O][=C][C][=C][C][=C][Branch1][C][O][C][Branch1][Ring1][O][C][=C][Ring1][=Branch2][C][O][C][=C][C][Branch1][Ring1][C][=O][=C][C][=C][Ring1][Branch2][O]");
        check_split("[C][C][C][=Branch1][C][=C][C][=C][C][=N+1][Ring1][Branch1][C][=C][C][=C][Ring1][=Branch1][NH1][C][=C][Ring1][Branch1][C][=C][C][=C][Ring1][=Branch1][C][C][C][=C][N+1][=C][C][=C][C][=C][C][=C][C][=C][Ring1][=Branch1][NH1][C][Ring1][=Branch2][=C][Ring1][=N][C][=C][Ring1][P]");
        check_split("[C][N][C][C][C][C@H1][Ring1][Branch1][C][=C][C][=C][N][=C][Ring1][=Branch1]");
        check_split(
            r"[C][C][C][C@@H1][Branch1][C][O][C][C][\\C][=C][\\C][=C][\\C][#C][C][#C][\\C][=C][\\C][O][C][C][C][C@@H1][Branch1][C][O][C][C][/C][=C][/C][=C][/C][#C][C][#C][/C][=C][/C][O]",
        );
        check_split(
            r"[C][C][=C][Branch2][Ring2][Ring2][C][=Branch1][C][=O][C][C@@H1][Ring1][=Branch1][O][C][=Branch1][C][=O][C@@H1][C@H1][Branch1][Branch2][C][Ring1][Ring1][Branch1][C][C][C][/C][=C][Branch1][C][\\C][/C][=Branch1][C][=O][O][C][C][/C][=C][\\C][=C]",
        );
        check_split("[O][C][=C][C@H1][Branch1][Branch1][C@H1][Ring1][Branch1][O][C][=C][Ring1][Ring1][C][=C][Branch1][Ring1][O][C][C][=C][Ring1][Branch2][O][C][=Branch1][C][=O][C][=C][Ring1][#Branch1][C][C][C][=Branch1][C][=O][Ring1][Branch1]");
        check_split("[O][C][C@@H1][Branch1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@@H1][Branch1][C][O][C@@H1][Branch1][C][O][Ring1][Branch2]");
        check_split("[O][C][C@@H1][Branch1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@@H1][C@@H1][Ring1][#Branch1][C][=C][Branch1][C][O][C][Branch1][Ring1][O][C][=C][Branch1][C][O][C][=C][Ring1][#Branch2][C][=Branch1][C][=O][O][Ring1][#C]");
        check_split(
            r"[C][C][=Branch1][C][=O][O][C][C][C][Branch1][C][/C][=C][\\C][C@H1][Branch1][=Branch1][C][Branch1][C][C][=C][C][C][C][=C]",
        );
        check_split(
            "[C][C][C@H1][Branch1][C][O][C][C][C@@][Ring1][Ring2][C][C][C][O][Ring1][Branch1]",
        );
        check_split("[C][C][Branch1][C][C][C@@][C][C@@H1][Ring1][Ring1][C@@H1][Branch1][C][C][C][=Branch1][C][=O][C][Ring1][Branch2]");
        check_split("[O][C][C][C][=C][Branch1][C][C][N+1][=Branch1][Branch1][=C][S][Ring1][=Branch1][C][C][=C][N][=C][Branch1][C][C][N][=C][Ring1][#Branch1][N]");
    }
}