import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inspector"))

from lib import redact_pulumi_output  # noqa: E402


def test_redacts_db_admin_password_yaml_style():
    raw = '  + db_admin_password: "wMP3K7UvnAIsMO8yDXnkwDFO"'
    assert redact_pulumi_output(raw) == '  + db_admin_password: "[secret]"'


def test_redacts_administrator_login_password():
    raw = "administrator_login_password: 's3cr3t'"
    assert redact_pulumi_output(raw) == "administrator_login_password: '[secret]'"


def test_leaves_non_secret_output_unchanged():
    raw = '  + db_fqdn: "sc-e16dsv5-pg18-c100.postgres.database.azure.com"'
    assert redact_pulumi_output(raw) == raw


if __name__ == "__main__":
    test_redacts_db_admin_password_yaml_style()
    test_redacts_administrator_login_password()
    test_leaves_non_secret_output_unchanged()
    print("ok")
