{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE LambdaCase #-}

import Control.Lens hiding ((<.>), (??))
import Control.Lens.Internal.Zoom
import Control.Monad.State.Lazy
import Control.DeepSeq
import qualified GHC.Generics as GHC
import Data.Default
import Data.List.Split
import Data.Random.Normal
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.HMatrix
import System.IO
import System.Random

----------

data Dist = NormalDist (R, R)   -- distributions to generate examples
          | UniformDist (R, R)
  deriving (Show, Read, Eq, GHC.Generic)

instance NFData Dist

data NL = SoftmaxNL   -- NL stands for nonlinearity
        | SigmoidNL
        | IdNL
        | TanhNL
        | SoftplusNL
        | ReluNL
  deriving (Show, Read, Eq, Enum, GHC.Generic)

instance NFData NL

data Config = Config { _epsilonInit :: R
                     , _learningRate :: R
                     , _regParameter :: R
                     , _numIterations :: Int
                     , _numExamples :: Int
                     , _coordDist :: Dist
                     , _numDisplayedDigits :: Int
                     , _topOutput :: (Int, NL)
                     , _hiddenOutputs :: [(Int, NL)]
                     , _numInputs :: Int
                     }
  deriving (Show, Read, Eq, GHC.Generic)
makeLenses ''Config

instance Default Config where
  def = Config { _epsilonInit = 0.1
               , _learningRate = 3.0
               , _regParameter = 0.1 
               , _numIterations = 100
               , _numExamples = 1000
               , _coordDist = UniformDist (-1, 1)
               , _numDisplayedDigits = 16
               , _topOutput = (4, SoftmaxNL)
               , _hiddenOutputs = [ (32, TanhNL) -- closest to top
                                  , (32, TanhNL)
                                  ]
               , _numInputs = 2
               }

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
                   , _emitNL :: NL
                   , _absorbNL :: NL
                   }
  deriving (Show, Read, Eq, GHC.Generic)
makeLenses ''Layer

instance NFData Layer

data NN = NN { _topNN :: Layer   -- NN stands for neural network
             , _mHiddenNN :: Maybe NN
             }
  deriving (Show, Read, Eq, GHC.Generic)
makeLenses ''NN

instance NFData NN

----------

createLayer :: (NL, NL) -> Matrix R -> Layer
createLayer (nlOut, nlIn) mat =
  let (m, n) = size mat
   in Layer { _weights = mat
            , _gradients = Grad { _accGrad = konst 0 (m, n)
                                , _counterGrad = 0
                                }
            , _activities = konst 0 m
            , _deviations = konst 0 (n - 1)
            , _emitNL = nlOut
            , _absorbNL = nlIn
            }

createNN :: (NL, Matrix R) -> [(NL, Matrix R)] -> NN
createNN (nlTop, mTop) = \case
  [] -> NN { _topNN = createLayer (nlTop, IdNL) mTop
           , _mHiddenNN = Nothing
           }
  (nl1, mat1) : zs ->
    NN { _topNN = createLayer (nlTop, nl1) mTop
       , _mHiddenNN = Just $ createNN (nl1, mat1) zs
       }

----------

newRandomMatrix :: (R, R) -> (Int, Int) -> IO (Matrix R)
newRandomMatrix valRange (numRows, numCols) = do 
  g <- newStdGen
  return $
    matrix numCols $
    take (numRows * numCols) $
    randomRs valRange g

initMatrices :: (R, R) -> (Int, [Int]) -> Int
      -> IO (Matrix R, [Matrix R])
initMatrices r (nOut, nHUs) nIn = case nHUs of 
  [] -> do
    t <- newRandomMatrix r (nOut, nIn + 1)
    return $ (t, [])
  nHU : nHUs' -> do
    t <- newRandomMatrix r (nOut, nHU + 1)
    (t1, ts) <- initMatrices r (nHU, nHUs') nIn
    return $ (t, t1 : ts)

----------

initNN :: Config -> IO NN
initNN c = do
  (t, ms) <- initMatrices (-eps, eps) (nOut, nHUs) nIn
  return $ createNN (nlOut, t) (zip nlHUs ms)
    where
      eps = c ^. epsilonInit
      (nOut, nlOut) = c ^. topOutput
      (nHUs, nlHUs) = unzip $ c ^. hiddenOutputs
      nIn = c ^. numInputs 

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

sigmoid :: R -> R
sigmoid z = 1 / (1 + exp (-z))

softplus :: R -> R
softplus z = log (1 + (exp z))

relu :: R -> R
relu x = if (x > 0) then x else 0

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

forwardSigmoid :: (MonadState Layer m) => Vector R -> m (Vector R)
forwardSigmoid x = do
  a <- forwardLinear x
  return $ cmap sigmoid a

forwardId :: (MonadState Layer m) => Vector R -> m (Vector R)
forwardId x = do
  a <- forwardLinear x
  return $ a   -- apply identity transformation

forwardTanh :: (MonadState Layer m) => Vector R -> m (Vector R)
forwardTanh x = do
  a <- forwardLinear x
  return $ cmap tanh a 

forwardSoftplus :: (MonadState Layer m) => Vector R -> m (Vector R)
forwardSoftplus x = do
  a <- forwardLinear x
  return $ cmap softplus a

forwardRelu :: (MonadState Layer m) => Vector R -> m (Vector R)
forwardRelu x = do
  a <- forwardLinear x
  return $ cmap relu a

forwardNL :: (MonadState Layer m) => NL -> Vector R -> m (Vector R)
forwardNL = \case
  SoftmaxNL -> forwardSoftmax
  SigmoidNL -> forwardSigmoid
  IdNL -> forwardId
  TanhNL -> forwardTanh
  SoftplusNL -> forwardSoftplus
  ReluNL -> forwardRelu

---

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

backwardSigmoid :: (MonadState Layer m) => Vector R -> m (Vector R)
backwardSigmoid delta = do
  d <- backwardLinear delta
  a <- use activities
--  let aa = diag (a * (1 - a))
--  return $ app aa d
  return $ (a * (1 - a)) * d

backwardId :: (MonadState Layer m) => Vector R -> m (Vector R)
backwardId delta = do 
  d <- backwardLinear delta
  return $ d   -- multiply by identity matrix

backwardTanh :: (MonadState Layer m) => Vector R -> m (Vector R)
backwardTanh delta = do 
  d <- backwardLinear delta
  a <- use activities
  return $ (1 - (a * a)) * d 

backwardSoftplus :: (MonadState Layer m) => Vector R -> m (Vector R)
backwardSoftplus delta = do
  d <- backwardLinear delta
  a <- use activities
  return $ (1 - exp (- a)) * d 

backwardRelu :: (MonadState Layer m) => Vector R -> m (Vector R)
backwardRelu delta = do
  d <- backwardLinear delta
  a <- use activities
  return $ (cmap (\z -> if (z > 0) then 1 else 0) a) * d

backwardNL :: (MonadState Layer m) => NL -> Vector R -> m (Vector R)
backwardNL = \case
  SoftmaxNL -> backwardSoftmax
  SigmoidNL -> backwardSigmoid
  IdNL -> backwardId
  TanhNL -> backwardTanh
  SoftplusNL -> backwardSoftplus
  ReluNL -> backwardRelu

---

forwardNN :: (Functor (Zoomed m1 (Vector R)),
      Applicative (Zoomed m (Vector R)), Zoom m m NN NN,
      Zoom m1 m Layer NN) =>
     Vector R -> m (Vector R)
forwardNN x = do
  t <- use topNN
  mh <- use mHiddenNN
  case mh of
    Nothing -> zoom topNN $ forwardNL (t ^. emitNL) x
    Just _ -> do
      a <- zoom (mHiddenNN . _Just) $ forwardNN x
      b <- zoom topNN $ forwardNL (t ^. emitNL) a
      return $ b

backwardNN :: (Functor (Zoomed m (Vector R)),
      Applicative (Zoomed m1 (Vector R)), Zoom m m1 Layer NN,
      Zoom m1 m1 NN NN) =>
     Vector R -> m1 (Vector R)
backwardNN delta = do
  t <- use topNN
  mh <- use mHiddenNN
  case mh of
    Nothing -> zoom topNN $ backwardNL (t ^. absorbNL) delta
    Just _ -> do
      dtop <- zoom topNN $ backwardNL (t ^. absorbNL) delta
      d <- zoom (mHiddenNN . _Just) $ backwardNN dtop
      return $ d

propagateNN :: (Functor (Zoomed m1 (Vector R)),
      Applicative (Zoomed m (Vector R)), Zoom m m NN NN,
      Zoom m1 m Layer NN) =>
     (Vector R, Vector R) -> m (Vector R)
propagateNN (x, y) = do
  a <- forwardNN x
  d <- backwardNN (a - y)
  return $ d

---

resetLayerActivities :: (MonadState Layer m) => m ()
resetLayerActivities = do
  n <- fmap size $ use activities
  activities .= konst 0 n 

resetLayerDeviations :: (MonadState Layer m) => m ()
resetLayerDeviations = do
  n <- fmap size $ use deviations
  deviations .= konst 0 n 

resetLayerGradients :: (Functor (Zoomed m ()),
      Zoom m n Grad Layer) => n ()
resetLayerGradients = do
  zoom gradients $ do
    sz <- fmap size $ use accGrad
    accGrad .= konst 0 sz
    counterGrad .= 0

resetLayerADG :: (Functor (Zoomed m1 ()), Zoom m1 m Grad Layer) => m ()
resetLayerADG = do 
  resetLayerGradients
  resetLayerDeviations
  resetLayerActivities

resetADGNN :: (Functor (Zoomed m1 ()), Functor (Zoomed m2 ()),
      Applicative (Zoomed m3 ()), Zoom m1 m Layer NN,
      Zoom m2 m1 Grad Layer, Zoom m3 m NN NN, m3 ~ m) =>
     m ()
resetADGNN = do
  zoom topNN $ resetLayerADG
  mh <- use mHiddenNN
  case mh of
    Nothing -> return ()
    Just _ -> zoom (mHiddenNN . _Just) $ resetADGNN

---

updateLayerGradients :: (Functor (Zoomed m1 ()), 
      Zoom m1 m Grad Layer) => m ()
updateLayerGradients = do
  delta <- use deviations
  a <- use activities
  let da = mul (asColumn delta) (asRow (vcons 1 a))
  zoom gradients $ do
    accGrad %= (\m -> add m da)
    counterGrad += 1

updateGradientsNN :: (Functor (Zoomed m1 ()), Functor (Zoomed m2 ()),
      Applicative (Zoomed m3 ()), Zoom m1 m Layer NN,
      Zoom m2 m1 Grad Layer, Zoom m3 m NN NN, m3 ~ m) =>
     m ()
updateGradientsNN = do
  zoom topNN $ updateLayerGradients
  mh <- use mHiddenNN
  case mh of
    Nothing -> return ()
    Just _ -> zoom (mHiddenNN . _Just) $ updateGradientsNN

learnExampleNN :: (Functor (Zoomed m1 ()),
      Functor (Zoomed m1 (Vector R)),
      Functor (Zoomed m2 ()), Applicative (Zoomed m ()),
      Applicative (Zoomed m (Vector R)), Zoom m m NN NN,
      Zoom m1 m Layer NN, Zoom m2 m1 Grad Layer) =>
     (Vector R, Vector R) -> m (Vector R)
learnExampleNN e = do
  d <- propagateNN e
  deepseq d (return ())   -- avoid a memory leak
  updateGradientsNN
  return $ d

learnNN :: (Functor (Zoomed m1 ()), Functor (Zoomed m1 (Vector R)),
      Functor (Zoomed m2 ()), Applicative (Zoomed m ()),
      Applicative (Zoomed m (Vector R)), Traversable t, Zoom m m NN NN,
      Zoom m1 m Layer NN, Zoom m2 m1 Grad Layer) =>
     t (Vector R, Vector R) -> m (t (Vector R))
learnNN = mapMOf traverse $ learnExampleNN

---

updateLayerWeights :: (Functor (Zoomed m1 ()), 
      Zoom m1 m Grad Layer) => Config -> m ()
updateLayerWeights c = do 
  let alpha = c ^. learningRate
  let lambda = c ^. regParameter
  gn <- use gradients
  let g = gn ^. accGrad
  let n = gn ^. counterGrad
  let coeff = (- alpha) / ((fromInteger . fromIntegral) n)
  weights %= (\m -> m * (scalar (1 + lambda * coeff)) + g * (scalar coeff))
  resetLayerGradients

updateWeightsNN :: (Functor (Zoomed m1 ()),
      Functor (Zoomed m2 ()),
      Applicative (Zoomed m3 ()), Zoom m1 m Layer NN,
      Zoom m2 m1 Grad Layer, Zoom m3 m NN NN, m3 ~ m) =>
     Config -> m ()
updateWeightsNN c = do
  zoom topNN $ updateLayerWeights c
  mh <- use mHiddenNN
  case mh of
    Nothing -> return ()
    Just _ -> zoom (mHiddenNN . _Just) $ updateWeightsNN c

trainOnceNN :: (Functor (Zoomed m1 ()),
      Functor (Zoomed m1 (Vector R)),
      Functor (Zoomed m2 ()), Applicative (Zoomed m ()),
      Applicative (Zoomed m (Vector R)), Traversable t, Zoom m m NN NN,
      Zoom m1 m Layer NN, Zoom m2 m1 Grad Layer,
      NFData (t (Vector R))) =>
     Config -> t (Vector R, Vector R) -> m (t (Vector R))
trainOnceNN c es = do
  ds <- learnNN es
  deepseq ds (return ())   -- avoid a memory leak
  updateWeightsNN c
  return $ ds

trainNN :: (Functor (Zoomed m1 ()),
      Functor (Zoomed m1 (Vector R)),
      Functor (Zoomed m2 ()), Applicative (Zoomed m ()),
      Applicative (Zoomed m (Vector R)), Traversable t,
      Zoom m1 m Layer NN, Zoom m2 m1 Grad Layer, Zoom m m NN NN,
      NFData (t (Vector R))) =>
     Config -> t (Vector R, Vector R) -> m [t (Vector R)]
trainNN c es = foldlMOf folded f [] [1..n]
  where
    f ts _ = do
      t <- trainOnceNN c es
      deepseq t (return ())   -- avoid a memory leak
      return $ (t : ts)
    n = c ^. numIterations

---

trainedNN :: (Monad m, Traversable t, NFData (t (Vector R))) =>
     Config -> t (Vector R, Vector R) -> NN -> m NN
trainedNN c es s = return $ (flip execState) s $
  trainNN c es >> resetADGNN

{-
-- A similar version below has a memory leak problem: <--- check this
trainedNN' :: (Monad m, Traversable t, NFData (t (Vector R))) =>
     Config -> t (Vector R, Vector R) -> NN -> m NN
trainedNN' c es s = (flip execStateT) s $ trainNN c es >> resetADGNN
-}

---

predictNN :: NN -> Vector R -> Vector R
predictNN nn = (flip evalState) nn . forwardNN

----------

newNormalRandomVectors :: (R, R) -> Int -> Int -> IO [Vector R]
newNormalRandomVectors (mu, sigma) n d = do
  g <- newStdGen
  return $ fmap vector $ take n $ chunksOf d $ normals' (mu, sigma) g


newUniformRandomVectors :: (R, R) -> Int -> Int -> IO [Vector R]
newUniformRandomVectors r n d = do
  g <- newStdGen
  return $ fmap vector $ take n $ chunksOf d $ randomRs r g


newRandomVectors :: Config -> Int -> Int -> IO [Vector R]
newRandomVectors c n d = case (c ^. coordDist) of
  NormalDist (mu, sigma) -> newNormalRandomVectors (mu, sigma) n d
  UniformDist (r1, r2) -> newUniformRandomVectors (r1, r2) n d


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
  vs <- newRandomVectors c n 2
  return $ zip vs (fmap quadrantVec vs)
    where
      n = c ^. numExamples

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
  disp ndd . asRow $ predictNN nn (vector [x0, x1])
  putStrLn ""
    where
      ndd = c ^. numDisplayedDigits

dispParameters :: Config -> IO ()
dispParameters c = do
  putStrLn $ "Top output: " ++ show (c ^. topOutput)
  putStrLn $ "Hidden outputs: " ++ show (c ^. hiddenOutputs)
  putStrLn $ "Number of inputs: " ++ show (c ^. numInputs)
  putStrLn $ "Number of examples: " ++ show (c ^. numExamples)
  putStrLn $ "Coordinate distribution: " ++ show (c ^. coordDist)
  putStrLn $ "Number of iterations: " ++ show (c ^. numIterations)
  putStrLn $ "Learning rate: " ++ show (c ^. learningRate) 
  putStrLn $ "Regularization parameter: " ++ show (c ^. regParameter)
  putStrLn $ "Initialization noise level: " ++ show (c ^. epsilonInit)

----------

main :: IO ()
main = do
  let cfg = def

  putStrLn $ underline "Neural network configuration:"
  dispParameters cfg
  putStrLn ""

  putStr "Training the neural network ... " >> hFlush stdout
  nnInit <- initNN cfg
  es <- newRandomQuadrantExamples cfg
  nnTrained <- trainedNN cfg es nnInit
  deepseq nnTrained (return ())
  putStrLn "Done!\n"

  putStrLn $ underline "Predictions for the test points:"
  ts <- testPoints
  mapM_ (dispTestPoint cfg nnTrained) ts

  return ()

----------
