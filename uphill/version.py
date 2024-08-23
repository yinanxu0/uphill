import os
from packaging.version import parse


from .logx import loggerx


def get_package_version():
    version_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "VERSION")
    version = '0.0.0'
    for content in open(version_file, 'r', encoding='utf8').readlines():
        content = content.strip()
        if len(content) > 0:
            version = content
            break
    return version


def check_package_version(version: str='0.0.0'):
    _version = get_package_version()
    source_version = parse(version)
    package_version = parse(_version)
    if source_version.major < package_version.major:
        loggerx.warning(f"UpHill={_version} try to load data saved by lower major version({version})")
        return True
    elif source_version.major > package_version.major:
        loggerx.critical(f"UpHill={_version} cannot load data saved by higher major version({version})")
        return False
    else:
        if source_version.minor != package_version.minor:
            loggerx.warning(f"UpHill={_version} try to load data saved by different minor version({version})")
        return True

