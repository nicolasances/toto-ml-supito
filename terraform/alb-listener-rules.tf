##############################################################
# 1. Target Groups
#    This section creates the Target Group for this service.
##############################################################
resource "aws_lb_target_group" "service_tg" {
  name = format("%s-tg-%s", local.toto_microservice_name, var.toto_environment)
  port = 8080
  protocol = "HTTP"
  vpc_id = var.toto_vpc_id
  target_type = "ip"
}

##############################################################
# 2. ALB Listener Rules
##############################################################
resource "aws_lb_listener_rule" "alb_listener_rule" {
  listener_arn = var.alb_listener_arn
  condition {
    http_header {
      http_header_name = "toto-service"
      values = [ local.toto_microservice_name ]
    }
  }
  action {
    type = "forward"
    target_group_arn = aws_lb_target_group.service_tg.arn
  }
  tags = {
    Name = format("%s-rule", local.toto_microservice_name)
  }
}