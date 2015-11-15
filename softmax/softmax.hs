{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DeriveGeneric #-}

import Control.Lens hiding ((<.>), (??))
import Control.Lens.Internal.Zoom
import Control.Monad.State.Lazy
import Control.DeepSeq
import qualified GHC.Generics as GHC
import Data.Default
import Data.List.Split
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.HMatrix
import System.IO
import System.Random

----------

data Config = Config { _epsilonInit :: R
                     , _learningRate :: R
                     , _numIterations :: Int
                     , _numInputs :: Int
                     , _numHiddenUnits :: Int
                     , _numOutputs :: Int
                     , _numExamples :: Int
                     , _coordRange :: (R, R)
                     , _numDisplayedDigits :: Int
                     }
  deriving (Show, Read, Eq, GHC.Generic)
makeLenses ''Config

instance Default Config where
  def = Config { _epsilonInit = 0.01
               , _learningRate = 10.0
               , _numIterations = 300
               , _numInputs = 2
               , _numHiddenUnits = 32
               , _numOutputs = 4
               , _numExamples = 1000
               , _coordRange = (-1, 1)
               , _numDisplayedDigits = 8
               }

----------

data Grad = Grad { _accGrad :: Matrix R  -- accumulate sum
                 , _counterGrad :: Int   -- count updates
                 }
  deriving (Show, Read, Eq, GHC.Generic)
makeLenses ''Grad

instance NFData Grad

data Layer = Layer { _weights :: Matrix R
                   , _gradients :: Grad
                   , _activities :: Vector R
                   , _deviations :: Vector R
                   }
  deriving (Show, Read, Eq, GHC.Generic)
makeLenses ''Layer
-- A layer remembers activities and deviations that it *receives*. 
-- What it *gives* is put in the monadic return values. 

instance NFData Layer

data NN = NN { _layer1 :: Layer
             , _layer2 :: Layer
             }
  deriving (Show, Read, Eq, GHC.Generic)
makeLenses ''NN

instance NFData NN

----------

vcons :: R -> Vector R -> Vector R
vcons x v = vjoin [vector [x], v]

vsum :: Vector R -> R
vsum v = (konst 1.0 (size v)) <.> v

softmax :: Vector R -> Vector R
softmax v =
  let mv = maxElement v
      u = cmap (\x -> exp (x - mv)) v
      z = vsum u
   in u / (scalar z)

----------

forwardLinear :: (MonadState Layer m) => Vector R -> m (Vector R)
forwardLinear x = do 
  activities .= x
  theta <- use weights
  return $ app theta (vcons 1.0 x)

forwardSoftmax :: (MonadState Layer m) => Vector R -> m (Vector R)
forwardSoftmax x = do
  a <- forwardLinear x
  return $ softmax a

forward :: ( Functor (Zoomed m1 (Vector R))
           , Zoom m1 m Layer NN
           )
        => Vector R
        -> m (Vector R)
forward x = do
  a1 <- zoom layer1 $ forwardSoftmax x
  a2 <- zoom layer2 $ forwardSoftmax a1
  return $ a2

backwardLinear :: (MonadState Layer m) => Vector R -> m (Vector R)
backwardLinear delta = do 
  deviations .= delta
  theta <- use weights
  return $ app (tr (theta ?? (All, Drop 1))) delta 

backwardSoftmax :: (MonadState Layer m) => Vector R -> m (Vector R)
backwardSoftmax delta = do
  d <- backwardLinear delta
  a <- use activities
  let aa = diag a - mul (asColumn a) (asRow a)
  return $ app aa d

backward :: ( Functor (Zoomed m1 (Vector R))
            , Zoom m1 m Layer NN
            )
         => Vector R
         -> m (Vector R)
backward delta = do
  d2 <- zoom layer2 $ backwardSoftmax delta
  d1 <- zoom layer1 $ backwardSoftmax d2
  return $ d1

propagate :: ( Functor (Zoomed m1 (Vector R))
             , Zoom m1 m Layer NN
             )
          => (Vector R, Vector R)
          -> m (Vector R)
propagate (x, y) = do
  a <- forward x
  d <- backward (a - y)
  return $ d

updateLayerGradients :: ( Functor (Zoomed m0 ())
                        , Zoom m0 m Grad Layer
                        )
                     => m ()
updateLayerGradients = do
  delta <- use deviations
  a <- use activities
  let da = mul (asColumn delta) (asRow (vcons 1 a))
  zoom gradients $ do
    accGrad %= (\m -> add m da)
    counterGrad += 1

resetLayerGradients :: ( Functor (Zoomed m0 ())
                       , Zoom m0 m Grad Layer
                       )
                    => m ()
resetLayerGradients = do
  zoom gradients $ do
    sz <- fmap size $ use accGrad
    accGrad .= konst 0 sz
    counterGrad .= 0

resetLayerActivities :: (MonadState Layer m) => m ()
resetLayerActivities = do
  n <- fmap size $ use activities
  activities .= konst 0 n 

resetLayerDeviations :: (MonadState Layer m) => m ()
resetLayerDeviations = do
  n <- fmap size $ use deviations
  deviations .= konst 0 n 

resetLayerADG :: ( Functor (Zoomed m0 ())
                 , Zoom m0 m Grad Layer
                 )
              => m ()
resetLayerADG = do 
  resetLayerGradients
  resetLayerDeviations
  resetLayerActivities

resetADG :: ( Functor (Zoomed m ())
           , Functor (Zoomed m0 ())
           , Zoom m n Layer NN
           , Zoom m0 m Grad Layer
           )
        => n ()
resetADG = do
  zoom layer1 $ resetLayerADG
  zoom layer2 $ resetLayerADG

updateLayerWeights :: ( Functor (Zoomed m0 ())
                      , Zoom m0 m Grad Layer
                      )
                   => Config
                   -> m ()
updateLayerWeights c = do 
  let alpha = c ^. learningRate
  gn <- use gradients
  let g = gn ^. accGrad
  let n = gn ^. counterGrad
  let coeff = (- alpha) / ((fromInteger . fromIntegral) n)
  weights %= (\m -> m + g * (scalar coeff))
  resetLayerGradients

learnExample :: ( Functor (Zoomed m1 ())
                , Functor (Zoomed m1 (Vector R))
                , Functor (Zoomed m0 ())
                , Zoom m1 m Layer NN
                , Zoom m0 m1 Grad Layer
                )
             => (Vector R, Vector R)
             -> m (Vector R)
learnExample e = do
  d <- propagate e
  deepseq d (return ())   -- avoid a memory leak
  zoom layer1 $ updateLayerGradients
  zoom layer2 $ updateLayerGradients
  return $ d

learn :: ( Functor (Zoomed m1 ())
         , Functor (Zoomed m1 (Vector R))
         , Functor (Zoomed m0 ())
         , Traversable t
         , Zoom m1 m Layer NN
         , Zoom m0 m1 Grad Layer
         )
      => t (Vector R, Vector R)
      -> m (t (Vector R))
learn = mapMOf traverse $ learnExample

trainOnce :: ( Functor (Zoomed m1 ())
             , Functor (Zoomed m1 (Vector R))
             , Functor (Zoomed m0 ())
             , t ~ []
             , Zoom m1 m Layer NN
             , Zoom m0 m1 Grad Layer
             )
          => Config
          -> t (Vector R, Vector R)
          -> m (t (Vector R))
trainOnce c es = do
  ds <- learn es
  deepseq ds (return ())   -- avoid a memory leak
  zoom layer1 $ updateLayerWeights c
  zoom layer2 $ updateLayerWeights c
  return $ ds

train :: ( Functor (Zoomed m1 ())
         , Functor (Zoomed m1 (Vector R))
         , Functor (Zoomed m0 ())
         , t ~ []
         , Zoom m1 m Layer NN
         , Zoom m0 m1 Grad Layer
         )
      => Config
      -> t (Vector R, Vector R)
      -> m [t (Vector R)]
train c es = foldlMOf folded f [] [1..n]
  where
    f ts _ = do
      t <- trainOnce c es
      deepseq t (return ())   -- avoid a memory leak
      return $ (t : ts)
    n = c ^. numIterations

trainedNN :: (t ~ [], Monad m) => Config -> t (Vector R, Vector R) -> NN -> m NN
trainedNN c es s = return $ (flip execState) s $ train c es >> resetADG

{-
-- A similar version below has a memory leak problem: 
trainedNN :: (t ~ [], Monad m) => Config -> [(Vector R, Vector R)] -> NN -> m NN
trainedNN c es s = (flip execStateT) s $ train c es >> resetADG
-}

predict :: NN -> Vector R -> Vector R
predict nn = (flip evalState) nn . forward

----------

layerFromMatrix :: Matrix R -> Layer
layerFromMatrix mat =
  let (m, n) = size mat
   in Layer { _weights = mat
            , _gradients = Grad { _accGrad = konst 0 (m, n)
                                , _counterGrad = 0
                                }
            , _activities = konst 0 (n - 1)
            , _deviations = konst 0 (n - 1)
            }

nnFromMatrices :: (Matrix R, Matrix R) -> NN
nnFromMatrices (mat2, mat1) =
  NN { _layer1 = layerFromMatrix mat1
     , _layer2 = layerFromMatrix mat2
     }

----------

newRandomMatrix :: (R, R) -> (Int, Int) -> IO (Matrix R)
newRandomMatrix valRange (numRows, numCols) = do 
  g <- newStdGen
  return $
    matrix numCols $
    take (numRows * numCols) $
    randomRs valRange g

newRandomVectors :: (R, R) -> Int -> Int -> IO [Vector R]
newRandomVectors r n d = do
  g <- newStdGen
  return $ fmap vector $ take n $ chunksOf d $ randomRs r g

quadrantVec :: Vector R -> Vector R
quadrantVec v = vector $
  if (x > 0.0)
  then
    if (y > 0.0)
    then [1.0, 0.0, 0.0, 0.0]
    else [0.0, 0.0, 0.0, 1.0]
  else
    if (y > 0.0)
    then [0.0, 1.0, 0.0, 0.0]
    else [0.0, 0.0, 1.0, 0.0]
    where x = v ! 0
          y = v ! 1

newRandomQuadrantExamples :: Config -> IO ([(Vector R, Vector R)])
newRandomQuadrantExamples c = do
  vs <- newRandomVectors r n 2
  return $ zip vs (fmap quadrantVec vs)
    where
      n = c ^. numExamples
      r = c ^. coordRange


testPoints :: (Monad m) => m ([(R, R)])
testPoints = return $ [ ( 0.5,  0.5)
                      , (-0.5,  0.5)
                      , (-0.5, -0.5)
                      , ( 0.5, -0.5)
                      ]
----------

underline :: String -> String
underline xs = xs ++ "\n" ++ (replicate (length xs) '-')

dispTestPoint :: Config -> NN -> (R, R) -> IO ()
dispTestPoint c nn (x0, x1) = do 
  putStr $ (show (x0, x1)) ++ " " ++ (show $ quadrantVec (vector [x0, x1])) ++ ": "
  disp ndd . asRow $ predict nn (vector [x0, x1])
  putStrLn ""
    where
      ndd = c ^. numDisplayedDigits

dispParameters :: Config -> IO ()
dispParameters c = do
  putStrLn $ "Number of inputs: " ++ show (c ^. numInputs)
  putStrLn $ "Number of hidden units: " ++ show (c ^. numHiddenUnits)
  putStrLn $ "Number of outputs: " ++ show (c ^. numOutputs)
  putStrLn $ "Number of examples: " ++ show (c ^. numExamples)
  putStrLn $ "Number of iterations: " ++ show (c ^. numIterations)
  putStrLn $ "Learning rate: " ++ show (c ^. learningRate)

-- An initialization has values in IO monad.
initNN :: Config -> IO NN
initNN c = do
  mat1 <- newRandomMatrix (-eps, eps) (nHU, nIn + 1)  
  mat2 <- newRandomMatrix (-eps, eps) (nOut, nHU + 1)
  return $ nnFromMatrices (mat2, mat1)
    where 
      eps = c ^. epsilonInit
      nIn = c ^. numInputs
      nHU = c ^. numHiddenUnits
      nOut = c ^. numOutputs

----------

main :: IO ()
main = do
  putStrLn $ underline "Softmax neural network with one hidden layer."
  dispParameters def
  putStrLn ""

  putStr "Training the neural network ... " >> hFlush stdout
  nnInit <- initNN def
  es <- newRandomQuadrantExamples def
  nnTrained <- trainedNN def es nnInit
  deepseq nnTrained (return ())
  putStrLn "Done!\n"

  putStrLn $ underline "Predictions for the test points:"
  ts <- testPoints
  mapM_ (dispTestPoint def nnTrained) ts

  return ()

----------
