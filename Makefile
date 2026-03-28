# Helper Makerfile for development, testing, and packaging
# Run `make help` for more information

# Configurable variables
PYTHON                      ?= python3
CANN_VERSION                ?= 8.5.0
CHIP_TYPE                   ?= A3
ARCH                        := $(shell uname -i)
SUDO                        := $(shell command -v sudo >/dev/null 2>&1 && echo sudo || echo)
TOOLKIT_URL                 := https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20$(CANN_VERSION)/Ascend-cann-toolkit_$(CANN_VERSION)_linux-$(ARCH).run
CANN_TOOLKIT                := Ascend-cann-toolkit.run
CANN_KERNELS 			    := Ascend-cann-kernels.run
REQ_RT_FILE				    := requirements.txt
REQ_DEV_FILE				:= requirements-dev.txt
REQ_RT_STAMP                := .req_rt_installed
REQ_DEV_STAMP               := .req_dev_installed
REQ_RT_HASH                 := $(shell md5sum $(REQ_RT_FILE) 2>/dev/null | cut -d ' ' -f 1)
REQ_DEV_HASH                := $(shell md5sum $(REQ_DEV_FILE) 2>/dev/null | cut -d ' ' -f 1)


# =====================
# CANN Kernel URL
# =====================
IS_8_5_0                    := $(filter 8.5.0, $(CANN_VERSION))
CANN_BASE_URL 	            := https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20$(CANN_VERSION)
ifeq ($(IS_8_5_0), $(CANN_VERSION)) # VERSION 8.5.0: Unified naming convention
	KERNEL_URL := $(CANN_BASE_URL)/Ascend-cann-$(CHIP_TYPE)-ops_$(CANN_VERSION)_linux-$(ARCH).run
else
	KERNEL_URL:= $(CANN_BASE_URL)/Ascend-cann-toolkit_$(CANN_VERSION)_linux-$(ARCH).run
endif


.PHONY: check_cann_url
check_cann_url: ## Check if CANN Toolkit and Kernel URLs are accessible
	@echo "Checking CANN Toolkit URL: $(TOOLKIT_URL)"
	@curl -s --head $(TOOLKIT_URL) | head -n 1 | grep "200 OK" > /dev/null || (echo "Error: CANN Toolkit URL is not accessible." && exit 1)
	@echo "CANN Toolkit URL is valid."
	@echo "Checking CANN Kernel URL: $(KERNEL_URL)"
	@curl -s --head $(KERNEL_URL) | head -n 1 | grep "200 OK" > /dev/null || (echo "Error: CANN Kernel URL is not accessible." && exit 1)
	@echo "CANN Kernel URL is valid."


# =====================
# CANN Installation
# =====================
.PHONY: install_cann
install_cann: $(CANN_TOOLKIT) $(CANN_KERNELS) ## Download and install CANN
	chmod +x $^
	./$(CANN_TOOLKIT) --full --quiet
	./$(CANN_KERNELS) --install --quiet

$(CANN_TOOLKIT):
	@echo "Downloading CANN Toolkit from: $(TOOLKIT_URL)"
	@curl -sSL "$(TOOLKIT_URL)" -o $@

$(CANN_KERNELS):
	@echo "Downloading CANN Kernels from: $(KERNEL_URL)"
	@curl -sSL "$(KERNEL_URL)" -o $@


# =====================
# Environment Setup
# =====================
.PHONY: install_deps
install_deps: ## Install OS-level dependencies
	@echo "Installing OS-level dependencies..."
	$(SUDO) apt-get update
	$(SUDO) apt-get install -yes --no-install-recommends \
		ca-certificates ccache clang ninja-build libzstd-dev \
		lld git python3 python3-dev python3-pip zlib1g-dev cmake
	@python3 -m pip install ninja


# =====================
# Python Dependencies
# =====================
.PHONY: install_dev_reqs
install_dev_reqs: ## Install TileLang development requirements
	@if [ ! -f $(REQ_DEV_STAMP) ] || [ "$(shell cat $(REQ_DEV_STAMP))" != "$(REQ_DEV_HASH)" ]; then \
		echo "Installing development requirements..."; \
		$(PYTHON) -m pip install -r $(REQ_DEV_FILE); \
		echo "$(REQ_DEV_HASH)" > $(REQ_DEV_STAMP); \
	else \
		echo "Development requirements are already up to date."; \
	fi

.PHONY: install_rt_reqs
install_rt_reqs: ## Install TileLang runtime requirements
	@if [ ! -f $(REQ_RT_STAMP) ] || [ "$(shell cat $(REQ_RT_STAMP))" != "$(REQ_RT_HASH)" ]; then \
		echo "Installing runtime requirements..."; \
		$(PYTHON) -m pip install -r $(REQ_RT_FILE); \
		echo "$(REQ_RT_HASH)" > $(REQ_RT_STAMP); \
	else \
		echo "Runtime requirements are already up to date."; \
	fi


# =====================
# HELP
# =====================
.PHONY: help
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v '^_' | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
