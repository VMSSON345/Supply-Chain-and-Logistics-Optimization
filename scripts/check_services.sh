#!/bin/bash

echo "=================================================="
echo "  Checking Service Health"
echo "=================================================="

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check Kafka
echo -e "\n${YELLOW}Checking Kafka...${NC}"
if docker exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092 &> /dev/null; then
    echo -e "${GREEN}✓ Kafka is healthy${NC}"
else
    echo -e "${RED}✗ Kafka is not responding${NC}"
fi

# Check Elasticsearch
echo -e "\n${YELLOW}Checking Elasticsearch...${NC}"
if curl -s http://localhost:9200/_cluster/health | grep -q "green\|yellow"; then
    echo -e "${GREEN}✓ Elasticsearch is healthy${NC}"
    curl -s http://localhost:9200/_cluster/health | jq
else
    echo -e "${RED}✗ Elasticsearch is not responding${NC}"
fi

# Check Spark
echo -e "\n${YELLOW}Checking Spark Master...${NC}"
if curl -s http://localhost:8080 | grep -q "Spark Master"; then
    echo -e "${GREEN}✓ Spark Master is healthy${NC}"
else
    echo -e "${RED}✗ Spark Master is not responding${NC}"
fi

# Check Kibana
echo -e "\n${YELLOW}Checking Kibana...${NC}"
if curl -s http://localhost:5601/api/status | grep -q "available"; then
    echo -e "${GREEN}✓ Kibana is healthy${NC}"
else
    echo -e "${RED}✗ Kibana is not responding${NC}"
fi

echo -e "\n${GREEN}=================================================="
echo -e "  Health Check Complete"
echo -e "==================================================${NC}"
