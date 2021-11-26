# help及び.PHONYの記法に関してはこちらの記事を参考にさせて頂いています： https://qiita.com/shakiyam/items/cdd3c11eba978202a628
export UID=$(shell id -u)
include .env
export

dc := docker-compose -f docker/docker-compose.yml 

ifeq ($(DEVICE), gpu)
	dc += -f docker/docker-compose.gpu.yml
else ifeq ($(DEVICE), cpu)
else
	dc := $(error invalid DEVICE environment variable: '$(DEVICE)')
endif

dc-exec-root := $(dc) exec -u root app

ifeq ($(UID), 0)
	export BUILD_TARGET=root
	dc-exec := $(dc-exec-root)
else
	export BUILD_TARGET=non-root
	dc-exec := $(dc) exec -u user app
endif

.DEFAULT_GOAL := help

# all targets are phony
.PHONY: $(shell egrep -o ^[a-zA-Z_-]+: Makefile| sed 's/://')

help: ## print this help
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: build up chown-mounted-dirs pipenv-install  ## docker実行環境のsetup

chown-mounted-dirs: ## rootユーザで作成されたdirectoryの所有者をuserに変更
	@if [ ! "$(UID)" = "0" ]; then\
		$(dc-exec-root) chown -R user .venv;\
		$(dc-exec-root) chown -R user logs;\
		$(dc-exec-root) chown -R user models;\
		$(dc-exec-root) chown -R user dataset;\
	fi

build: ## docker-compose build
	$(dc) build

pipenv-install:  ## container内部でpipenv installを実行 
	$(dc-exec) pipenv install --dev --deploy

up:  ## docker-compose up
	$(dc) up -d

down:  ## docker-compose down
	$(dc) down

erase:  ## docker-compose down and remove volumes
	$(dc) down -v

stop:  ## docker-compose stop
	$(dc) stop

start:  ## docker-compose start
	$(dc) start

sh:  ## hostのcurrent userと同じUIDを持つuserとしてcontainerに入る
	$(dc-exec) /bin/bash

sh-root:  ## root userとしてcontainerに入る
	$(dc-exec-root) /bin/bash

jupyter:  ## container上でjupyter notebookを立ち上げる
	$(dc-exec) pipenv run jupyter

install-vscode-extensions-container: JQ_KEY='.recommendations|join(" ")'
install-vscode-extensions-container:  ## VSCodeのcontainer開発環境へ推奨のpluginをインストールする（containerにattachしたVSCode上のshellでのみ利用可能）
	$(foreach extension, $(shell jq -r $(JQ_KEY) .vscode/extensions.container.json),  code --install-extension $(extension))

install-vscode-extensions-host: JQ_KEY='.recommendations|join(" ")'
install-vscode-extensions-host:  ## hostのVSCodeへ推奨のpluginをインストールする（host上で起動したVSCodeのshellでのみ利用可能）
	$(foreach extension, $(shell jq -r $(JQ_KEY) .vscode/extensions.host.json),  code --install-extension $(extension))
