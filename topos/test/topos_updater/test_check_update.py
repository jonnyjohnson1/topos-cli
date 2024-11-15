from ...utils.check_for_update import check_for_update, update_topos

import logging
logging.basicConfig(level=logging.DEBUG)

update_is_available = check_for_update("jonnyjohnson1", "topos-cli")
print(update_is_available)