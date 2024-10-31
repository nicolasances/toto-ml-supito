########################################################
# 1. Task Definition
########################################################
resource "aws_ecs_task_definition" "service_task_def" {
  family = local.toto_microservice_name
  requires_compatibilities = ["FARGATE"]
  execution_role_arn = var.ecs_execution_role_arn
  task_role_arn = var.ecs_task_role_arn
  cpu = 1024
  memory = 2048
  network_mode = "awsvpc"
  container_definitions = jsonencode([
    {
      name      = local.toto_microservice_name
      image     = format("%s.dkr.ecr.eu-west-1.amazonaws.com/%s:%s", var.aws_account_id, local.toto_microservice_name, var.container_image_tag)
      cpu       = 1024
      memory    = 2048
      essential = true
      portMappings = [
        {
          containerPort = 8080
          hostPort      = 8080
          protocol      = "tcp"
          appProtocol   = "http"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs", 
        options = {
          awslogs-create-group = "true"
          awslogs-group = format("/ecs/%s", local.toto_microservice_name)
          awslogs-region = "eu-west-1"
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

########################################################
# 2. Service
########################################################
resource "aws_ecs_service" "service" {
  name = local.toto_microservice_name
  cluster = var.ecs_cluster_arn
  task_definition = aws_ecs_task_definition.service_task_def.arn
  desired_count = 1
  capacity_provider_strategy {
    base = 0
    capacity_provider = "FARGATE"
    weight = 1
  }
  network_configuration {
    subnets = [ var.ecs_subnet_1, var.ecs_subnet_2 ]
    security_groups = [ var.ecs_security_group ]
    assign_public_ip = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.service_tg.arn
    container_name = local.toto_microservice_name
    container_port = 8080
  }
}