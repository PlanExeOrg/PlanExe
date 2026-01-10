"""
Locate PlanExe's config files, like .env and llm_config.json. The .env file is optional when the environment variables are provided by the host.

Finds config files by checking the following locations in order:
1. The directory specified by the PLANEXE_CONFIG_PATH environment variable. It must be an absolute path.
2. The current working directory (CWD).
3. The PlanExe project root directory (three levels above this file's location).

Usage: without any PLANEXE_CONFIG_PATH environment variable.
PROMPT> python -m worker_plan_api.planexe_config

Usage: with a PLANEXE_CONFIG_PATH environment variable set.
PROMPT> PLANEXE_CONFIG_PATH='/Users/neoneye/git/PlanExeGroup/PlanExe' python -m worker_plan_api.planexe_config

IDEA: validate the contents of ".env"
IDEA: validate the contents of "llm_config.json"
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, ClassVar
import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)

class ConfigNameEnum(str, Enum):
    DOTENV = ".env"
    LLM_CONFIG_JSON = "llm_config.json"

class PlanExeConfigError(Exception):
    """Raised when there is an error with the configuration."""
    pass

@dataclass
class PlanExeConfig:
    """
    Holds the resolved paths to PlanExe configuration files and the env var value used.
    
    Attributes:
        planexe_config_path: Optional[Path] - The directory specified by PLANEXE_CONFIG_PATH
        dotenv_path: Optional[Path] - Path to the .env file
        llm_config_json_path: Optional[Path] - Path to the llm_config.json file
    """
    planexe_config_path: Optional[Path]
    dotenv_path: Optional[Path]
    llm_config_json_path: Optional[Path]
    
    _instance: ClassVar[Optional['PlanExeConfig']] = None

    def raise_if_required_files_not_found(self) -> None:
        """
        Raises a PlanExeConfigError if required configuration files are not found.
        The .env file is optional (environment variables can be provided by the host).

        :raises: PlanExeConfigError if required files are not found
        """
        missing_files = []
        if self.llm_config_json_path is None:
            missing_files.append(ConfigNameEnum.LLM_CONFIG_JSON.value)
        
        if missing_files:
            msg = f"Required configuration file(s) not found: {', '.join(missing_files)}"
            logger.error(msg)
            raise PlanExeConfigError(msg)
        if self.dotenv_path is None:
            logger.info("Optional configuration file '.env' not found; relying on environment variables only.")
        # If no missing files, method completes silently.
    
    @classmethod
    def load(cls) -> 'PlanExeConfig':
        """
        Loads configuration paths by searching predefined locations.
        Implements a singleton pattern to avoid repeated filesystem scans.
        
        :return: An instance of PlanExeConfig with resolved paths.
        """
        if cls._instance is not None:
            return cls._instance

        logger.debug("PlanExeConfig.load() creating a new instance...")
        planexe_config_path = cls.resolve_planexe_config_path()
        dotenv_path = cls.find_file_in_search_order(ConfigNameEnum.DOTENV.value, planexe_config_path, is_optional=True)
        llm_config_json_path = cls.find_file_in_search_order(ConfigNameEnum.LLM_CONFIG_JSON.value, planexe_config_path)
        cls._instance = cls(
            planexe_config_path=planexe_config_path,
            dotenv_path=dotenv_path,
            llm_config_json_path=llm_config_json_path
        )
        return cls._instance

    @classmethod
    def resolve_planexe_config_path(cls) -> Optional[Path]:
        """
        Resolves and validates the PLANEXE_CONFIG_PATH environment variable.
        It's expected to be an absolute path to a directory.
        
        :return: A Path object if valid, otherwise None.
        """
        path_str = os.environ.get("PLANEXE_CONFIG_PATH")
        if path_str is None:
            logger.debug("PLANEXE_CONFIG_PATH is not set")
            return None
            
        try:
            path_obj = Path(path_str)
        except Exception as e: # If path_str is bizarre
            logger.error(f"Invalid PLANEXE_CONFIG_PATH string '{path_str!r}': {e!r}")
            return None
        if not path_obj.is_absolute():
            logger.error(f"PLANEXE_CONFIG_PATH must be an absolute path: {path_obj!r}")
            return None
        if not path_obj.is_dir():
            logger.error(f"PLANEXE_CONFIG_PATH must be a directory: {path_obj!r}")
            return None
        logger.debug(f"Using PLANEXE_CONFIG_PATH: {path_obj!r}")
        return path_obj

    @classmethod
    def find_file_in_search_order(cls, filename: str, planexe_config_path: Optional[Path], is_optional: bool = False) -> Optional[Path]:
        """
        Finds a specific configuration file based on a precedence of locations.

        Search order:
        1. Directory from validated PLANEXE_CONFIG_PATH (if provided and valid).
        2. Current Working Directory (CWD).
        3. PlanExe project root.

        :param filename: The name of the file to find (e.g., ".env").
        :param planexe_config_path: The validated absolute directory path from PLANEXE_CONFIG_PATH.
        :param is_optional: When True, missing file is logged at INFO instead of WARNING.
        :return: The Path to the file if found, otherwise None.
        """
        # Step 1: Check if PLANEXE_CONFIG_PATH is set and contains the file
        if planexe_config_path is not None:
            config_file_path = planexe_config_path / filename
            if config_file_path.is_file():
                logger.debug(f"Found {filename!r} at config_file_path: {config_file_path!r}")
                return config_file_path

        # Step 2: Check if file exists in current working directory
        cwd_file_path = Path.cwd() / filename
        if cwd_file_path.is_file():
            logger.debug(f"Found {filename!r} at cwd_file_path: {cwd_file_path!r}")
            return cwd_file_path

        # Step 3: Check if file exists in PlanExe root directory
        # This file is at: worker_plan/worker_plan_api/planexe_config.py
        # So we need 3 .parent calls to reach the PlanExe root.
        root_file_path = Path(__file__).parent.parent.parent / filename
        if root_file_path.is_file():
            logger.debug(f"Found {filename!r} at root_file_path: {root_file_path!r}")
            return root_file_path

        if is_optional:
            logger.info(f"Optional file {filename!r} not found in any of the search locations (ENV_VAR, CWD, Project Root).")
        else:
            logger.warning(f"{filename!r} not found in any of the search locations (ENV_VAR, CWD, Project Root).")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    config = PlanExeConfig.load()
    print(f"config: {config!r}")
    config.raise_if_required_files_not_found()
    
