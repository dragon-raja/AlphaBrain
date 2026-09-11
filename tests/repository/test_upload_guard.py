import subprocess

from tools.repository.check import staged_check


def test_staged_guard_rejects_credentials_without_printing_values(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    token = "ghp_" + "A" * 40  # Synthetic fixture, not a credential.
    (tmp_path / ".env").write_text("TOKEN=" + token)
    subprocess.run(["git", "add", ".env"], cwd=tmp_path, check=True)
    result = staged_check(tmp_path)
    assert result["status"] == "FAIL"
    assert any("Sensitive path" in value for value in result["errors"])
    assert any("credential" in value for value in result["errors"])
    assert token not in str(result)
