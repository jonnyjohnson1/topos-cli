{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.apacheKafka
    pkgs.poetry
  ];

  shellHook = ''
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

    sleep 3 &

    poetry install
    cd chat_server
    # poetry run fastapi dev
    poetry run python -m app.app
  '';
}