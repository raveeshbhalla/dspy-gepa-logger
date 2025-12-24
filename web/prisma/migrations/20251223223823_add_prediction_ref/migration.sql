-- AlterTable
ALTER TABLE "Evaluation" ADD COLUMN "predictionRef" TEXT;

-- AlterTable
ALTER TABLE "Run" ADD COLUMN "valsetExampleIds" TEXT;
