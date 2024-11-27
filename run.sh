#!/bin/bash

# Function to check if docker-compose is available
has_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        return 0
    elif command -v docker compose &> /dev/null; then
        return 2
    else
        return 1
    fi
}

# Function to run docker commands
run_docker_command() {
    local cmd=$1
    shift  # Remove first argument (cmd)
    
    if has_docker_compose; then
        case "$cmd" in
            "build")
                docker-compose build
                ;;
            "run")
                docker-compose run trading-bot $@
                ;;
            "logs")
                docker-compose logs $@
                ;;
            "down")
                docker-compose down
                ;;
        esac
    elif [ $? -eq 2 ]; then
        case "$cmd" in
            "build")
                docker compose build
                ;;
            "run")
                docker compose run trading-bot $@
                ;;
            "logs")
                docker compose logs $@
                ;;
            "down")
                docker compose down
                ;;
        esac
    else
        case "$cmd" in
            "build")
                docker build -t trading-bot .
                ;;
            "run")
                docker run -v $(pwd)/data/trades:/app/data/trades \
                          -v $(pwd)/data/logs:/app/data/logs \
                          -e TZ=America/Los_Angeles \
                          -e LOG_DIR=/app/data/logs \
                          -e TRADES_DIR=/app/data/trades \
                          trading-bot --asset-type $2 --tickers ${@:3}
                ;;
            "logs")
                docker logs trading-bot $@
                ;;
            "down")
                docker stop trading-bot
                ;;
        esac
    fi
}

# Function to setup environment
setup_environment() {
    echo "Setting up trading bot environment..."
    
    # Create data directories
    mkdir -p data/trades data/logs
    chmod -R 777 data/trades data/logs
    
    echo "Created data directories:"
    echo "  - data/trades (for trade history)"
    echo "  - data/logs (for log files)"
    
    # Build docker image
    echo "Building Docker image..."
    run_docker_command build
}

# Function to display help
show_help() {
    echo "Robinhood Trading Bot"
    echo "Usage: $0 {setup|start|logs|analyze|stop|help}"
    echo ""
    echo "Commands:"
    echo "  setup              Setup environment and build Docker image"
    echo "  start [options]    Start the trading bot"
    echo "  logs [type]        View logs (trades/bot/docker)"
    echo "  analyze [date]     Analyze trades for a specific date"
    echo "  stop               Stop the trading bot"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 start crypto --debug --tickers BTC ETH DOGE"
    echo "  $0 start stock --debug --tickers AAPL MSFT GOOGL"
    echo "  $0 logs trades    # View trade files"
    echo "  $0 logs bot      # View bot logs"
    echo "  $0 logs          # View docker logs"
    echo "  $0 analyze 20240315"
}

# Main script
case "$1" in
    "setup")
        setup_environment
        ;;
    
    "start")
        # Check if environment is set up
        if [ ! -d "data/trades" ] || [ ! -d "data/logs" ]; then
            echo "Environment not set up. Running setup first..."
            setup_environment
        fi
        
        asset_type=$2
        shift 2  # Remove 'start' and asset_type from arguments
        
        if [ "$asset_type" != "stock" ] && [ "$asset_type" != "crypto" ]; then
            echo "Error: Asset type must be 'stock' or 'crypto'"
            exit 1
        fi
        
        # Start the bot with all remaining arguments
        run_docker_command run --asset-type $asset_type $@
        ;;
    
    "logs")
        case "$2" in
            "trades")
                ls -l data/trades/
                ;;
            "bot")
                tail -f data/logs/trading_log.txt
                ;;
            *)
                run_docker_command logs -f
                ;;
        esac
        ;;
    
    "analyze")
        if [ -z "$2" ]; then
            # Use today's date if none specified
            date=$(date +%Y%m%d)
        else
            date=$2
        fi
        run_docker_command run --analyze-date $date
        ;;
    
    "stop")
        run_docker_command down
        echo "Trading bot stopped"
        ;;
    
    "help")
        show_help
        ;;
    
    *)
        show_help
        exit 1
        ;;
esac 