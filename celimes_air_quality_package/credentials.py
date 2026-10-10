"""Credential setup for the data services the package downloads from.

Nothing in this package stores keys in code. Each downloader takes the key
as an argument, and if you leave it out it reads the environment variable
listed below, so notebooks never need to contain a key:

    AIRNOW_API_KEY        AirNow API key
    AQS_EMAIL, AQS_KEY    EPA AQS (AirData) API credentials
    LAADS_TOKEN           NASA LAADS DAAC bearer token (MODIS)
    CDS_API_KEY           Copernicus CDS personal access token (ERA5)

TEMPO downloads use NASA Earthdata login from ~/.netrc
(run get_earthdata_credentials() once).
"""

import netrc
import os
from pathlib import Path


def env_or_arg(value, env_name, what):
    """Return value if given, else the environment variable, else raise."""
    if value:
        return value
    value = os.environ.get(env_name)
    if not value:
        raise ValueError(
            f"No {what} provided. Pass it as an argument or set the "
            f"{env_name} environment variable."
        )
    return value


# ------------------------------------------------------------------
# Copernicus CDS (ERA5)
# ------------------------------------------------------------------

def write_cdsapirc(api_key):
    cdsapirc = Path.home() / ".cdsapirc"
    cdsapirc.write_text(f"url: https://cds.climate.copernicus.eu/api\nkey: {api_key}\n")
    return cdsapirc


def check_cds_credentials():
    if not (Path.home() / ".cdsapirc").exists():
        raise FileNotFoundError(
            "CDS API credentials not found. Run get_era5_credentials(), pass api_key=, "
            "or set CDS_API_KEY.\n"
            "  1. Create a free account at https://cds.climate.copernicus.eu\n"
            "  2. Copy your Personal Access Token from https://cds.climate.copernicus.eu/profile"
        )


def get_era5_credentials():
    """Prompt for a CDS Personal Access Token and save it to ~/.cdsapirc (run once)."""
    cdsapirc = Path.home() / ".cdsapirc"
    print("\n--- ERA5 CDS API Credentials Setup ---")
    print("Sign up at https://cds.climate.copernicus.eu, then copy your token from")
    print("https://cds.climate.copernicus.eu/profile\n")
    if cdsapirc.exists():
        if input(".cdsapirc already exists. Overwrite? (yes/no): ").strip().lower() != "yes":
            print("Keeping existing credentials.")
            return
    token = input("Enter your CDS Personal Access Token: ").strip()
    if not token:
        raise ValueError("Token cannot be empty.")
    print(f"Credentials saved to {write_cdsapirc(token)}")


# ------------------------------------------------------------------
# NASA Earthdata (TEMPO)
# ------------------------------------------------------------------

EARTHDATA_HOST = "urs.earthdata.nasa.gov"


def check_earthdata_credentials():
    """Raise a clear error if ~/.netrc has no Earthdata login."""
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        raise FileNotFoundError(
            "Earthdata credentials not found. Run get_earthdata_credentials(), or create "
            f"{netrc_path} containing:\n\n"
            f"    machine {EARTHDATA_HOST}\n    login YOUR_USERNAME\n    password YOUR_PASSWORD\n\n"
            "Free account: https://urs.earthdata.nasa.gov/users/new"
        )
    try:
        auth = netrc.netrc(str(netrc_path)).authenticators(EARTHDATA_HOST)
    except (netrc.NetrcParseError, OSError) as exc:
        raise ValueError(f"Could not read {netrc_path}: {exc}") from exc
    if auth is None:
        raise ValueError(f"No entry for {EARTHDATA_HOST} in {netrc_path}")


def get_earthdata_credentials():
    """Prompt for NASA Earthdata username/password and save them to ~/.netrc (run once).

    Other machines already in ~/.netrc are kept.
    """
    import getpass

    netrc_path = Path.home() / ".netrc"
    print("\n--- NASA Earthdata Credentials Setup ---")
    print("Free account: https://urs.earthdata.nasa.gov/users/new\n")

    kept = []
    if netrc_path.exists():
        if input(".netrc exists. Replace the Earthdata entry? (yes/no): ").strip().lower() != "yes":
            print("Keeping existing credentials.")
            return
        # Keep every machine block except Earthdata's
        lines = netrc_path.read_text().splitlines()
        skip = False
        for line in lines:
            tok = line.split()
            if tok and tok[0] == "machine":
                skip = len(tok) > 1 and tok[1] == EARTHDATA_HOST
            if not skip:
                kept.append(line)

    username = input("Earthdata username: ").strip()
    password = getpass.getpass("Earthdata password: ").strip()
    if not username or not password:
        raise ValueError("Username and password cannot be empty.")

    content = "\n".join(kept).rstrip()
    content = (content + "\n" if content else "") + (
        f"machine {EARTHDATA_HOST}\n    login {username}\n    password {password}\n"
    )
    netrc_path.write_text(content)
    if os.name != "nt":
        os.chmod(netrc_path, 0o600)
    print(f"Credentials saved to {netrc_path}")
