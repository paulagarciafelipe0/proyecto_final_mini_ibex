provider "aws" {
  region = "eu-west-1"
}

# Bucket para código de lambdas
resource "aws_s3_bucket" "lambda_bucket" {
  bucket        = "mini-ibex-lambda-code"
  force_destroy = true

  lifecycle {
    prevent_destroy = false
  }
}

resource "aws_s3_bucket_public_access_block" "block_public_access" {
  bucket = aws_s3_bucket.lambda_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.lambda_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

# IAM role para Lambdas con permisos básicos + acceso DynamoDB
resource "aws_iam_role" "lambda_exec_role" {
  name = "lambda_execution_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_exec" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "lambda_dynamodb_access" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess"
}

# DynamoDB tables
resource "aws_dynamodb_table" "opciones" {
  name         = "OpcionesMiniIBEX"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "fecha_vencimiento"
  range_key    = "strike_tipo_timestamp"

  attribute {
    name = "fecha_vencimiento"
    type = "S"
  }

  attribute {
    name = "strike_tipo_timestamp"
    type = "S"
  }
}

resource "aws_dynamodb_table" "futuros" {
  name         = "FuturosMiniIBEX"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "fecha_vencimiento"

  attribute {
    name = "fecha_vencimiento"
    type = "S"
  }
}

resource "aws_dynamodb_table" "volatilidad" {
  name         = "VolatilidadMiniIBEX"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "fecha_vencimiento"
  range_key    = "strike_tipo_timestamp"

  attribute {
    name = "fecha_vencimiento"
    type = "S"
  }

  attribute {
    name = "strike_tipo_timestamp"
    type = "S"
  }
}

# Lambda: scraping_guardar_dynamo
resource "aws_lambda_function" "scraping_guardar_dynamo" {
  function_name = "scrapingGuardarDynamo"
  s3_bucket     = aws_s3_bucket.lambda_bucket.id
  s3_key        = "scraping_lambda_package.zip"
  handler       = "lambda_function_scraping.lambda_handler"
  runtime       = "python3.11"
  role          = aws_iam_role.lambda_exec_role.arn
  timeout       = 180
}

resource "aws_lambda_function" "calcular_volatilidad" {
  function_name = "calcularVolatilidadMiniIBEX"

  package_type = "Image"
  image_uri    = "786055265018.dkr.ecr.eu-west-1.amazonaws.com/lambda_volatilidad:2025-05-17"

  role         = aws_iam_role.lambda_exec_role.arn
  timeout      = 900
}


resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# EventBridge trigger diario para scraping
resource "aws_cloudwatch_event_rule" "daily_scraping_trigger" {
  name                = "daily-scraping-trigger"
  schedule_expression = "cron(0 17 * * ? *)"
}

resource "aws_cloudwatch_event_target" "scraping_lambda_target" {
  rule      = aws_cloudwatch_event_rule.daily_scraping_trigger.name
  target_id = "scraping-lambda"
  arn       = aws_lambda_function.scraping_guardar_dynamo.arn
}

resource "aws_lambda_permission" "allow_eventbridge_scraping" {
  statement_id  = "AllowExecutionFromEventBridgeScraping"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.scraping_guardar_dynamo.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.daily_scraping_trigger.arn
}

# Custom event pattern mejor para encadenar lambdas (recomendado)
resource "aws_cloudwatch_event_bus" "custom_bus" {
  name = "custom-mini-ibex-bus"
}

resource "aws_cloudwatch_event_rule" "scraping_to_volatility" {
  name        = "scraping-to-volatility"
  event_bus_name = aws_cloudwatch_event_bus.custom_bus.name
  event_pattern = jsonencode({
    "source": ["custom.miniibex.scraping"],
    "detail-type": ["Scraping Completed"]
  })
}

resource "aws_cloudwatch_event_target" "calculate_volatility_target" {
  rule      = aws_cloudwatch_event_rule.scraping_to_volatility.name
  target_id = "calculate-volatility"
  arn       = aws_lambda_function.calcular_volatilidad.arn
  event_bus_name = aws_cloudwatch_event_bus.custom_bus.name
}

resource "aws_lambda_permission" "allow_eventbridge_calculation" {
  statement_id  = "AllowExecutionFromEventBridgeCalculation"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.calcular_volatilidad.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.scraping_to_volatility.arn
}
