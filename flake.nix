{
  description = "topos";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    process-compose-flake.url = "github:Platonic-Systems/process-compose-flake";
    services-flake.url = "github:juspay/services-flake";
  };

  outputs = { self, nixpkgs, flake-parts, poetry2nix, process-compose-flake, services-flake }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
        imports = [ inputs.process-compose-flake.flakeModule ];
        systems = [ "x86_64-linux" "aarch64-darwin" ];
        perSystem = { self', pkgs, system, lib, ... }:
        let
            pkgs = import nixpkgs {
                inherit system;
                overlays = [
                inputs.poetry2nix.overlays.default
                (final: prev: {
                    toposPoetryEnv = final.callPackage toposPoetryEnv { };
                    pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
                    (python-final: python-prev: {
                        pystray = python-final.callPackage ./overlays/pystray/default.nix { };
                    })
                    ];
                })
                ];
            };

            # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
            #TODO: Figure out how to add setuptools to all the packages which need it, this is currently not working as expected.
            overrides = pkgs.poetry2nix.overrides.withDefaults (final: super:
            pkgs.lib.mapAttrs
                (attr: systems: super.${attr}.overridePythonAttrs
                (old: {
                    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ map (a: final.${a}) systems;
                }))
                {
                # https://github.com/nix-community/poetry2nix/blob/master/docs/edgecases.md#modulenotfounderror-no-module-named-packagename
                package = [ "setuptools" ];
                }
            );
            toposPoetryEnv = pkgs.poetry2nix.mkPoetryEnv {
            projectDir = self;
            preferWheels = true;
            inherit overrides;
            };

            envFile = pkgs.writeText "env_dev" (builtins.readFile ./.env_dev);
            parseEnvFile = envFile:
            let
                content = builtins.readFile envFile;
                lines = lib.filter (l: l != "" && !lib.hasPrefix "#" l) (lib.splitString "\n" content);
                parseLine = l:
                let
                    parts = lib.splitString "=" l;
                in
                    { name = lib.head parts; value = lib.concatStringsSep "=" (lib.tail parts); };
            in
                builtins.listToAttrs (map parseLine lines);
            envVars = parseEnvFile ./.env_dev;

            configFile = pkgs.copyPathToStore ./config.yaml;
            yq = pkgs.yq-go;

            # Note: This only loads the settings from the repos config file
            #        if one is not already set in the user's .config directory.
            toposSetupHook = ''
            export $(cat ${envFile} | xargs)
            export TOPOS_CONFIG_PATH="$HOME/.config/topos/config.yaml"
            mkdir -p "$(dirname "$TOPOS_CONFIG_PATH")"
            if [ ! -f "$TOPOS_CONFIG_PATH" ]; then
                echo "Creating new config file at $TOPOS_CONFIG_PATH"
                echo "# Topos Configuration" > "$TOPOS_CONFIG_PATH"
                ${yq}/bin/yq eval ${configFile} | while IFS= read -r line; do
                echo "$line" >> "$TOPOS_CONFIG_PATH"
                done
                echo "Config file created at $TOPOS_CONFIG_PATH"
            else
                echo "Config file already exists at $TOPOS_CONFIG_PATH"
            fi
            '';

            kafkaSetupHook = ''
            echo "Starting Kafka in Kraft mode..."

            # Set up necessary environment variables
            export KAFKA_HEAP_OPTS="-Xmx512M -Xms512M"
            export KAFKA_KRAFT_MODE=true
            echo ${pkgs.apacheKafka}

            # Prepare a default config for Kraft mode
            if [ ! -f ./kafka.properties ]; then
                echo "Initializing Kafka Kraft mode..."

                # Server 1 Kraft
                cp ${pkgs.apacheKafka}/config/kraft/server.properties ./server-1.properties
                sudo sed -i '57!s/PLAINTEXT/MQ/g' server-1.properties
                sudo sed -i '30s/.*/controller.quorum.voters=1@localhost:9091/' server-1.properties
                sudo sed -i '78s|.*|log.dirs=/tmp/kraft-combined-logs/server-1|' server-1.properties
                sudo sed -i '27s|.*|node.id=1|' server-1.properties
                sudo sed -i '42s|.*|listeners=MQ://:9092,CONTROLLER://:9091|' server-1.properties
                sudo sed -i '92s|.*|offsets.topic.replication.factor=1|' server-1.properties
                sudo sed -i '57s|.*|listener.security.protocol.map=CONTROLLER:PLAINTEXT,MQ:PLAINTEXT,SSL:SSL,SASL_PLAINTEXT:SASL_PLAINTEXT,SASL_SSL:SASL_SSL|' server-1.properties

            fi

            # Step 1
            KAFKA_CLUSTER_ID="$(${pkgs.apacheKafka}/bin/kafka-storage.sh random-uuid)"

            # Step 2
            ${pkgs.apacheKafka}/bin/kafka-storage.sh format -t $KAFKA_CLUSTER_ID -c ./server-1.properties

            # Step 3
            ${pkgs.apacheKafka}/bin/kafka-server-start.sh ./server-1.properties &

            # Step 4
            echo "Kafka environment is ready to use and running in detached terminals."

            # Step 5
            ${pkgs.apacheKafka}/bin/kafka-topics.sh --create --topic chat_topic --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

            sleep 3
            '';

        in
        {
            process-compose."services-flake-topos" = { config, ... }: {
                imports = [
                inputs.services-flake.processComposeModules.default
                (import ./topos-service.nix { inherit pkgs lib config; topos = self'.packages.topos; })
                ];
                services = let dataDirBase = "$HOME/.services-flake/llm"; in {
                    # Backend service to perform inference on LLM models
                    ollama."ollama1" = {
                        enable = true;

                        # The models are usually huge, downloading them in every project
                        # directory can lead to a lot of duplication. Change here to a
                        # directory where the Ollama models can be stored and shared across
                        # projects.
                        dataDir = "${dataDirBase}/ollama1";

                        # Define the models to download when our app starts
                        #
                        # You can also initialize this to empty list, and download the
                        # models manually in the UI.
                        #
                        # Search for the models here: https://ollama.com/library
                        models = [ "phi3" ];
                    };

                    postgres."pg1" = {
                      enable = true;
                      package = pkgs.postgresql_16.withPackages (p: [ p.pgvector ]);
                      port = 5432;
                      listen_addresses = "127.0.0.1";

                      initialDatabases = [
                        { name = "${envVars.POSTGRES_DB}"; }
                      ];
                      initialScript = {
                        before = ''
                          CREATE USER ${envVars.POSTGRES_USER} WITH SUPERUSER PASSWORD '${envVars.POSTGRES_PASSWORD}';
                        '';
                        after = ''
                          CREATE TABLE IF NOT EXISTS conversation_cache (
                              conv_id TEXT PRIMARY KEY,
                              message_data JSONB NOT NULL
                          );

                          CREATE TABLE IF NOT EXISTS entities (
                              id TEXT PRIMARY KEY,
                              label TEXT NOT NULL,
                              properties JSONB
                          );

                          CREATE TABLE IF NOT EXISTS relations (
                              source_id TEXT,
                              relation_type TEXT,
                              target_id TEXT,
                              properties JSONB,
                              PRIMARY KEY (source_id, relation_type, target_id)
                          );

                          GRANT ALL PRIVILEGES ON DATABASE ${envVars.POSTGRES_DB} TO ${envVars.POSTGRES_USER};
                          GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ${envVars.POSTGRES_USER};
                          GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ${envVars.POSTGRES_USER};
                          GRANT ALL PRIVILEGES ON SCHEMA public TO ${envVars.POSTGRES_USER};

                          ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ${envVars.POSTGRES_USER};
                          ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO ${envVars.POSTGRES_USER};

                          GRANT pg_read_all_data TO ${envVars.POSTGRES_USER};
                          GRANT pg_write_all_data TO ${envVars.POSTGRES_USER};
                        '';
                      };
                    };

                    topos.enable = true;
                    topos.args = [ "run" ];
                };
                settings.processes.topos.depends_on.pg1.condition = "process_healthy";
            };

            packages =  rec {
                toposPoetry = pkgs.poetry2nix.mkPoetryApplication {
                projectDir = self;
                preferWheels = true;
                inherit overrides;
                };
                topos = pkgs.writeShellScriptBin "topos" ''
                  ${toposSetupHook}
                  ${toposPoetry}/bin/topos "$@"
                '';
                default = self'.packages."services-flake-topos";
            };

            devShells = {
            # Shell for app dependencies.
            #
            #     nix develop
            #
            # Use this shell for developing your app.
            default = pkgs.mkShell {
                inputsFrom = [ toposPoetryEnv ];
                packages = [ ];
                shellHook = ''
                export PATH="${toposPoetryEnv}/bin:$PATH"
                ${toposSetupHook}
                '';
            };

            # Shell for poetry.
            #
            #     nix develop .#poetry
            #
            # Use this shell for changes to pyproject.toml and poetry.lock.
            poetry = pkgs.mkShell {
                packages = [ pkgs.poetry ];
            };
            };
            legacyPackages = pkgs;
        };
    };
}
